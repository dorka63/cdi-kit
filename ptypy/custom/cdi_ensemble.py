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
    """Run one member; no ensemble residual or ranking is calculated.

    Standard engines retain their own diagnostics inside Ptycho.run().
    """
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
    if not np.isfinite(result).all():
        raise ValueError("Reconstruction produced nonfinite values.")
    return result


def _initialize_worker(parameters):
    """Load already-preprocessed data once in each spawned process."""
    global _WORKER
    ptycho = Ptycho(parameters, level=2)
    pod = next(iter(ptycho.pods.values()))
    _WORKER = (ptycho, pod)


def _worker_task(index, field, support, chain):
    ptycho, pod = _WORKER
    return index, _execute_member(ptycho, pod, field, support, chain)


class CDIEnsemble:
    """Single-frame fixed-grid CDI ensemble with one workspace per worker.

    workers=1 uses the parent workspace. workers>1 lazily starts a persistent
    spawn-based process pool. No spatial alignment or field replacement by
    SVD is performed. Engine-local schedules restart for each block.
    Initial fields have random amplitude and zero phase inside the support.
    svd() uses all fields and stores mode zero as result; no member ranking.
    close() only releases resources. finalize() additionally saves result when
    io.rfile is configured. A normal context-manager exit calls finalize().
    No shrink-wrap implementation is included.
    """
    def __init__(self, parameters, seeds=range(8), workers=1, support_factory=None):
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
        requested_io = u.Param()
        if p.get("io") is not None:
            requested_io.update(p.io, in_place_depth=99)
        requested_io = requested_io.copy(depth=99)
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
        self._result_io = self.ptycho.p.io.copy(depth=99)
        self._result_io.update(requested_io, in_place_depth=99)
        self._result_io.rfile = requested_io.get("rfile", None)
        self._finalized = False
        self.result = None
        pods = list(self.ptycho.pods.values())
        if len(pods) != 1 or not pods[0].active:
            raise ValueError("Exactly one active POD is required.")

        self.pod = pods[0]
        self.pod.probe = np.ones(self.pod.probe.shape, dtype=self.ptycho.CType)
        if support_factory is not None:
            if not callable(support_factory):
                raise TypeError("support_factory must be callable.")
            set_cdi_support(self.ptycho, self.pod, support_factory(self.pod))
        self.support = cdi_support(self.ptycho, self.pod).copy()
        if self.support.shape != self.pod.object.shape:
            raise ValueError("Support and object shapes must match.")
        self.fields = np.empty(
            (len(self.seeds), *self.support.shape), dtype=self.ptycho.CType,
        )
        for index, seed in enumerate(self.seeds):
            rng = np.random.default_rng(seed)
            self.fields[index] = rng.random(size=self.support.shape) * self.support
        self.completed_stages = 0
        self.iterations_per_member = 0
        self._last_support = self.support.copy()
        self._result_support = None
        self._last_chain = self.engines.copy(depth=99)
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



    def run(self, engines=None):
        """Advance every field using the same chain; return self.

        This is one stage, not the end of the complete reconstruction.
        A new stage invalidates the previously computed primary SVD result.
        """
        if self._closed:
            raise RuntimeError("This ensemble is closed; construct another to resume execution.")
        chain = self._validate_engines(self.engines if engines is None else engines)
        if not chain:
            raise ValueError("Provide a nonempty engine chain.")
        updated = np.empty_like(self.fields)
        try:
            if self.workers == 1:
                for index, field in enumerate(self.fields):
                    updated[index] = _execute_member(
                        self.ptycho, self.pod, field, self.support, chain,
                    )
            else:
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
                    index, field = future.result()
                    updated[index] = field
        except BaseException:
            if self._pool is not None:
                self._pool.shutdown(wait=True, cancel_futures=True)
                self._pool = None
            raise
        finally:
            self._restore_workspace()
        self.fields = updated
        self.completed_stages += 1
        self.iterations_per_member += sum(pars.numiter for pars in chain.values())
        self._last_chain = chain.copy(depth=99)
        self._last_support = self.support.copy()
        self.result = None
        self._result_support = None
        self._restore_workspace()
        return self

    def svd(self, num_modes=1):
        """Use ALL fields, return (modes, full weights), and cache mode zero.

        No residuals, ranking, member rejection or spatial alignment.
        Calling svd() does not save files or change member fields.
        After editing fields directly, call svd() again before saving.
        """
        modes, weights = svd_modes(self.fields, num_modes=num_modes)
        self.result = modes[0].copy()
        self._result_support = self._last_support.copy()
        return modes, weights

    def set_support(self, support):
        support = np.asarray(support)
        if support.dtype != np.dtype(bool):
            raise TypeError("support must be a boolean array.")
        set_cdi_support(self.ptycho, self.pod, support)
        self.support = support.copy()

    def save_run(self, alt_file=None, kind=None, force_overwrite=True):
        """Save ONE primary field through the unmodified Ptycho.save_run().

        Ordinary results are stored in the standard object storage. The
        ensemble itself is not serialized, and support is not added to the
        file format. Final errors of a worker are not errors of an SVD mode.
        """
        if self.completed_stages == 0:
            raise RuntimeError("Run at least one reconstruction stage before saving.")
        if alt_file is None and not self._result_io.rfile:
            raise ValueError("Set io.rfile or provide alt_file to save_run().")
        save_kind = self._result_io.rformat if kind is None else kind
        if save_kind not in ("minimal", "dls"):
            raise ValueError("Ensemble results support rformat='minimal' or 'dls'.")
        if self.result is None:
            self.svd(num_modes=1)

        from ptypy.core.paths import Paths
        parent = self.ptycho
        old_object = self.pod.object.copy()
        old_probe = self.pod.probe.copy()
        old_exit = self.pod.exit.copy()
        old_support = cdi_support(parent, self.pod).copy()
        old_parameters = parent.p
        old_runtime = parent.runtime
        old_paths = parent.paths
        try:
            self.pod.object = self.result.astype(parent.CType, copy=True)
            self.pod.probe = np.ones_like(self.pod.probe)
            self.pod.exit = self.pod.probe * self.pod.object
            set_cdi_support(parent, self.pod, self._result_support)
            parent.p = old_parameters.copy(depth=99)
            parent.p.io = self._result_io.copy(depth=99)
            parent.p.engines = self._last_chain.copy(depth=99)
            # Preserve the resolved raw-input center in the standard parameters.
            scan = parent.model.scans[self.scan_label].ptyscan
            if scan.info.center is not None:
                parent.p.scans[self.scan_label].data.center = tuple(
                    float(value)
                    for value in scan.info.center
                )
            parent.runtime = old_runtime.copy(depth=99)
            parent.runtime.iter_info = []
            parent.runtime.engine = "CDIEnsemble"
            parent.runtime.iterations = self.iterations_per_member
            parent.runtime.cdi_result = "svd_mode_0"
            parent.paths = Paths(parent.p.io)
            parent.paths.runtime.engine = "CDIEnsemble"
            parent.paths.runtime.iteration = self.iterations_per_member
            parent.paths.runtime.iterations = self.iterations_per_member
            return parent.save_run(
                alt_file=None if alt_file is None else os.fspath(alt_file),
                kind=save_kind,
                force_overwrite=force_overwrite,
            )
        finally:
            self.pod.object = old_object
            self.pod.probe = old_probe
            self.pod.exit = old_exit
            set_cdi_support(parent, self.pod, old_support)
            parent.p = old_parameters
            parent.runtime = old_runtime
            parent.paths = old_paths

    def finalize(self):
        """Save the final primary mode if io.rfile is set, then close workers.

        Called on a normal context-manager exit, not between stages.
        A failed save is propagated; workers are still shut down.
        """
        if self._finalized:
            return
        try:
            if self.completed_stages and self._result_io.rfile:
                self.save_run()
        finally:
            self.close()
        self._finalized = True

    def close(self):
        """Release workers while retaining fields and SVD access."""
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.finalize()
        else:
            self.close()
        return False
