"""Fixed-grid CDI ensembles with optional persistent CPU process workers."""
from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.fft import set_workers
from ptypy import utils as u
from ptypy.core import Ptycho
from ptypy.engines import by_name
from ptypy.utils import parallel
from ptypy.custom.cdi_common import (
    CDIProjectionEngine, cdi_support, set_cdi_support,
    masked_amplitude_error, remove_global_phase,
)
import ptypy.custom.cdi_er
import ptypy.custom.cdi_hio
import ptypy.custom.cdi_raar

_WORKER = None
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)


def svd_modes(fields, num_modes=1):
    """Return RMS-scaled modes and full-spectrum weights without registration."""
    fields = np.asarray(fields, dtype=np.complex128)
    if fields.ndim != 3 or any(n == 0 for n in fields.shape):
        raise ValueError("fields must have nonempty (members, rows, columns) shape.")
    if not np.isfinite(fields).all():
        raise ValueError("fields must be finite.")
    limit = min(fields.shape[0], np.prod(fields.shape[1:]))
    if isinstance(num_modes, (bool, np.bool_)) or not isinstance(num_modes, (int, np.integer)) or not 1 <= num_modes <= limit:
        raise ValueError("Invalid num_modes.")
    scale = np.abs(fields).max()
    if scale == 0:
        raise ValueError("Cannot decompose an all-zero ensemble.")
    matrix = fields.reshape(len(fields), -1) / scale
    _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    weights = singular**2 / np.sum(singular**2)
    modes = (vh[:num_modes] * (singular[:num_modes, None] / np.sqrt(len(fields)))).reshape(
        num_modes, *fields.shape[1:]
    ) * scale
    for mode in modes:
        norm = np.linalg.norm(mode)
        if norm == 0:
            continue
        reference = mode.sum()
        if abs(reference) <= 1e-12 * norm * np.sqrt(mode.size):
            reference = mode.flat[np.argmax(np.abs(mode))]
        mode *= np.exp(-1j * np.angle(reference))
    return modes, weights


def _execute_member(ptycho, pod, field, support, chain):
    """Run a member in a process-local Ptycho workspace."""
    pod.object = field.copy()
    pod.probe = np.ones(pod.probe.shape, dtype=ptycho.CType)
    pod.exit = pod.probe * pod.object
    set_cdi_support(ptycho, pod, support)
    ptycho.runtime.iter_info = []
    with set_workers(1):
        for label, pars in chain.items():
            engine = by_name(pars.name)(ptycho, pars.copy(depth=99))
            engine.label = label
            ptycho.run(engine=engine)
    result = pod.object.copy()
    error = masked_amplitude_error(
        pod.fw(pod.probe * result), np.sqrt(pod.diff), pod.mask,
    )
    if not np.isfinite(result).all() or not np.isfinite(error):
        raise ValueError("Reconstruction produced nonfinite values.")
    return result, error


def _initialize_worker(parameters):
    """Load already-preprocessed data once in each spawned process."""
    global _WORKER
    ptycho = Ptycho(parameters, level=2)
    pod = next(iter(ptycho.pods.values()))
    _WORKER = (ptycho, pod)


def _worker_task(index, field, support, chain):
    ptycho, pod = _WORKER
    result, error = _execute_member(ptycho, pod, field, support, chain)
    return index, result, error


class CDIEnsemble:
    """Single-frame fixed-grid CDI ensemble with one workspace per worker.

    workers=1 uses the parent workspace. workers>1 lazily starts a persistent
    spawn-based process pool. No spatial alignment or field replacement by
    SVD is performed. Engine-local schedules restart for each block.
    close() releases the pool; member arrays remain available for analysis.
    """
    def __init__(self, parameters, seeds=range(8), workers=1):
        if parallel.size != 1:
            raise ValueError("MPI execution is not supported by CDIEnsemble.")
        self.seeds = tuple(seeds)
        if not self.seeds or any(isinstance(s, (bool, np.bool_)) or not isinstance(s, (int, np.integer)) or s < 0 for s in self.seeds):
            raise ValueError("Provide nonnegative integer seeds.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be distinct.")
        if isinstance(workers, (bool, np.bool_)) or not isinstance(workers, (int, np.integer)) or workers < 1:
            raise ValueError("workers must be a positive integer.")
        self.workers = min(int(workers), len(self.seeds))
        self._pool = None
        self._closed = False
        p = u.Param()
        p.update(parameters, in_place_depth=99)
        p = p.copy(depth=99)
        if len(p.scans) != 1:
            raise ValueError("Exactly one scan is required.")
        self.scan_label = next(iter(p.scans))
        scan = p.scans[self.scan_label]
        if scan.data.name != "CDIScan" or scan.get("resample", 1) != 1:
            raise ValueError("Provide one CDIScan with resample=1.")
        p.io = u.Param(rfile=None, autosave=u.Param(active=False),
                       autoplot=u.Param(active=False), interaction=u.Param(active=False))
        self.engines = self._validate_engines(p.get("engines", u.Param()))
        self.ptycho = Ptycho(p, level=2)
        pods = list(self.ptycho.pods.values())
        if len(pods) != 1 or not pods[0].active:
            raise ValueError("Exactly one active POD is required.")
        self.pod = pods[0]
        self.pod.probe = np.ones(self.pod.probe.shape, dtype=self.ptycho.CType)
        self.support = cdi_support(self.ptycho, self.pod).copy()
        if self.support.shape != self.pod.object.shape:
            raise ValueError("Support and object shapes must match.")
        self.amplitude = np.sqrt(self.pod.diff.copy())
        power = float(np.sum(self.pod.diff[self.pod.mask]))
        if not np.isfinite(power) or power <= 0:
            raise ValueError("Valid data must have positive finite total intensity.")
        level = np.sqrt(power / np.count_nonzero(self.support))
        self.fields = np.empty((len(self.seeds), *self.support.shape), dtype=self.ptycho.CType)
        for index, seed in enumerate(self.seeds):
            phase = np.random.default_rng(seed).uniform(-np.pi, np.pi, self.support.shape)
            self.fields[index] = remove_global_phase(
                level * self.support * np.exp(1j * phase), self.support,
            )
        self.completed_stages = 0
        self.iterations_per_member = 0
        self.selected_indices = None
        self.final_errors = self._errors(self.fields)
        self._restore_workspace()

        # Workers receive the final detector data, not the raw input TIFF.
        self._worker_parameters = p.copy(depth=99)
        self._worker_parameters.verbose_level = "error"
        self._worker_parameters.engines = u.Param()
        model = self.ptycho.model.scans[self.scan_label]
        geo = model.geometries[0]
        self._worker_parameters.scans[self.scan_label].data = u.Param(
            name="CDIScan", intensity=self.pod.diff.copy(), mask=self.pod.mask.copy(),
            energy=geo.energy, distance=geo.distance, psize=tuple(geo.psize),
            center=tuple(model.ptyscan.meta.center), autocenter=False,
            orientation=None, rebin=1, save=None,
            support=u.Param(kind="array", array=self.support.copy()),
            denoising=u.Param(active=False),
        )

    @staticmethod
    def _validate_engines(engines):
        result = u.Param()
        for label in sorted(engines):
            pars = u.Param()
            pars.update(engines[label], in_place_depth=99)
            pars = pars.copy(depth=99)
            cls = by_name(pars.name)
            if not issubclass(cls, CDIProjectionEngine):
                raise ValueError("Only CDIProjectionEngine subclasses are supported.")
            count = pars.get("numiter", cls.DEFAULT.numiter)
            if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 1:
                raise ValueError("numiter must be a positive integer.")
            pars.numiter = int(count)
            pars.numiter_contiguous = 1
            result[label] = pars
        return result

    def _restore_workspace(self):
        self.pod.object = self.fields[0].copy()
        self.pod.exit = self.pod.probe * self.pod.object
        set_cdi_support(self.ptycho, self.pod, self.support)

    def _errors(self, fields):
        return np.asarray([masked_amplitude_error(
            self.pod.fw(self.pod.probe * field), self.amplitude, self.pod.mask,
        ) for field in fields])

    def run(self, engines=None):
        """Advance all members, committing arrays only after full success."""
        if self._closed:
            raise RuntimeError("This ensemble is closed; construct another to resume execution.")
        chain = self._validate_engines(self.engines if engines is None else engines)
        if not chain:
            raise ValueError("Provide a nonempty engine chain.")
        updated = np.empty_like(self.fields)
        errors = np.empty(len(self.fields))
        try:
            if self.workers == 1:
                for index, field in enumerate(self.fields):
                    updated[index], errors[index] = _execute_member(
                        self.ptycho, self.pod, field, self.support, chain,
                    )
            else:
                # Spawned interpreters inherit limits before importing NumPy.
                old_env = {key: os.environ.get(key) for key in _THREAD_VARIABLES}
                try:
                    for key in _THREAD_VARIABLES:
                        os.environ[key] = "1"
                    if self._pool is None:
                        self._pool = ProcessPoolExecutor(
                            max_workers=self.workers, mp_context=mp.get_context("spawn"),
                            initializer=_initialize_worker,
                            initargs=(self._worker_parameters,),
                        )
                    futures = [self._pool.submit(
                        _worker_task, index, field.copy(), self.support.copy(), chain,
                    ) for index, field in enumerate(self.fields)]
                finally:
                    for key, value in old_env.items():
                        if value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = value
                for future in futures:
                    index, field, error = future.result()
                    updated[index] = field
                    errors[index] = error
        except BaseException:
            if self._pool is not None:
                self._pool.shutdown(wait=True, cancel_futures=True)
                self._pool = None
            raise
        finally:
            self._restore_workspace()
        self.fields = updated
        self.final_errors = errors
        self.completed_stages += 1
        self.iterations_per_member += sum(pars.numiter for pars in chain.values())
        self.selected_indices = None
        self._restore_workspace()
        return self

    def svd(self, keep=None, num_modes=1):
        count = len(self.fields) if keep is None else keep
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or not 1 <= count <= len(self.fields):
            raise ValueError("Invalid keep count.")
        errors = self._errors(self.fields)
        indices = np.argsort(errors, kind="stable")[:count]
        modes, weights = svd_modes(self.fields[indices], num_modes=num_modes)
        self.final_errors = errors
        self.selected_indices = indices.copy()
        return modes, weights

    def set_support(self, support):
        support = np.asarray(support)
        if support.dtype != np.dtype(bool):
            raise TypeError("support must be a boolean array.")
        set_cdi_support(self.ptycho, self.pod, support)
        self.support = support.copy()
        self.selected_indices = None

    def close(self):
        """Release workers while retaining fields and SVD access."""
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
