"""Four persistent CPU workers, random-amplitude starts, SVD and shrink-wrap.

Run from the repository root:
    python tutorial/cdi/experiment_nearfield.py

All settings and worker functions are in this file. No residual calculation,
residual-based ranking, per-iteration change norms or residual plots are used.
PtyPy prepares the data and geometry. The iteration kernel calls the installed
CDIHIO/CDIER object_update methods directly, bypassing engine_iterate metrics.
The main reconstruction for shrink-wrap is the first SVD mode of ALL members.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
from time import perf_counter, time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import set_workers
from scipy.ndimage import gaussian_filter, binary_closing

from ptypy import io, utils as u
from ptypy.core import Ptycho
from ptypy.engines import by_name
from ptypy.custom.cdi_common import (
    rectangular_support, cdi_support, masked_modulus_projection,
)
from ptypy.custom.cdi_ensemble import svd_modes
import ptypy.custom.cdi_er
import ptypy.custom.cdi_hio

IMAGE_PATH = Path(__file__).resolve().parent / "data" / "1.tif"
OUTPUT = Path("output/experiment_nearfield")

Z_UM = 6e5
LAMBDA_UM = 0.6328
PIXEL_SIZE_UM = (6.0, 6.0)
BETAS = (1.0, 0.7, 0.4, 0.1)
HIO_ITERATIONS = 30
ER_ITERATIONS = 10
INNER_REPEATS = 3

SEEDS = (0, 1, 2, 3)
WORKERS = 4
FFT_THREADS_PER_WORKER = 1
OUTER_CYCLES = 3
SVD_NUM_MODES = 4
RESET_PHASE = True

SIGMA_CLIP = 3.0
SIGMA_CLIP_MAX_ITER = 5
STATISTICS_MASK = None
SHRINKWRAP_SIGMA = 2.0
SHRINKWRAP_THRESHOLD = 0.1
SHRINKWRAP_CLOSING_ITERATIONS = 0

_WORKER = None


def load_data():
    intensity, _ = io.image_read(str(IMAGE_PATH))
    intensity = np.asarray(intensity, dtype=np.float64)
    if intensity.shape != (1030, 1288):
        raise ValueError(f"Expected image shape (1030, 1288), got {intensity.shape}.")
    size = min(intensity.shape)
    row_start = (intensity.shape[0] - size) // 2
    column_start = (intensity.shape[1] - size) // 2
    roi = (slice(row_start, row_start + size), slice(column_start, column_start + size))
    statistics_mask = STATISTICS_MASK
    if statistics_mask is not None:
        statistics_mask = np.asarray(statistics_mask)
        if statistics_mask.shape != intensity.shape:
            raise ValueError("STATISTICS_MASK must match the original TIFF shape.")
        if statistics_mask.dtype != np.dtype(bool):
            raise TypeError("STATISTICS_MASK must be a boolean array.")
        statistics_mask = statistics_mask[roi].copy()
    return intensity[roi].copy(), statistics_mask


def make_parameters(intensity, statistics_mask):
    support = rectangular_support(shape=intensity.shape, oversampling=2.0)
    p = u.Param(verbose_level="info", data_type="double")
    p.io = u.Param(rfile=None, autosave=u.Param(active=False),
                   autoplot=u.Param(active=False), interaction=u.Param(active=False))
    p.scans = u.Param()
    p.scans.cdi = u.Param(name="Vanilla", propagation="nearfield", ffttype="scipy",
                         sample=u.Param(fill=1.0 + 0.0j))
    p.scans.cdi.data = u.Param(
        name="CDIScan", intensity=intensity,
        energy=1.23984193e-9 / (LAMBDA_UM * 1e-6),
        distance=Z_UM * 1e-6,
        psize=tuple(value * 1e-6 for value in PIXEL_SIZE_UM),
        center=None, autocenter=True, autocenter_method="mass",
        autocenter_initial_center=None, autocenter_search_radius=14.0,
        autocenter_tolerance=1e-4, autocenter_min_overlap=0.5,
        rebin=1, orientation=None,
        support=u.Param(kind="array", array=support),
        denoising=u.Param(active=True, name="sigma_clip", sigma=SIGMA_CLIP,
                         max_iter=SIGMA_CLIP_MAX_ITER,
                         statistics_mask=statistics_mask, verbose=True),
    )
    p.engines = u.Param()
    return p


def worker_parameters(parent):
    """Reuse processed detector data and the actual parent geometry."""
    p = parent.p.copy(depth=99)
    p.verbose_level = "error"
    p.engines = u.Param()
    pod = next(iter(parent.pods.values()))
    model = parent.model.scans["cdi"]
    geo = model.geometries[0]
    p.scans.cdi.data = u.Param(
        name="CDIScan", intensity=pod.diff.copy(), mask=pod.mask.copy(),
        energy=geo.energy, distance=geo.distance, psize=tuple(geo.psize),
        center=tuple(model.ptyscan.meta.center), autocenter=False,
        rebin=1, orientation=None, save=None,
        support=u.Param(kind="array", array=cdi_support(parent, pod).copy()),
        denoising=u.Param(active=False),
    )
    return p


def initialize_worker(parameters):
    global _WORKER
    parent = Ptycho(parameters, level=2)
    pods = list(parent.pods.values())
    if len(pods) != 1 or not pods[0].active:
        raise ValueError("Exactly one active POD is required in each worker.")
    pod = pods[0]
    np.testing.assert_array_equal(pod.diff, parameters.scans.cdi.data.intensity)
    np.testing.assert_array_equal(pod.mask, parameters.scans.cdi.data.mask)
    np.testing.assert_allclose(
        parent.model.scans["cdi"].geometries[0].p.center,
        parameters.scans.cdi.data.center, rtol=0, atol=1e-10,
    )
    pod.probe = np.ones(pod.probe.shape, dtype=parent.CType)
    hio_updates = []
    for beta in BETAS:
        engine = by_name("CDIHIO")(parent, u.Param(beta=beta))
        hio_updates.append(engine.object_update)
    er_engine = by_name("CDIER")(parent, u.Param())
    _WORKER = dict(parent=parent, pod=pod, amplitude=np.sqrt(pod.diff),
                   hio_updates=hio_updates, er_update=er_engine.object_update)
    print(f"Worker ready: PID={os.getpid()}", flush=True)


def data_projection(field, pod, amplitude):
    detector_field = pod.fw(pod.probe * field)
    projected = masked_modulus_projection(detector_field, amplitude, pod.mask)
    return pod.bw(projected)


def run_member(index, seed, field, support, cycle):
    state = _WORKER
    pid = os.getpid()
    started = time()
    begin = perf_counter()
    print(f"Stage {cycle}: seed={seed}, PID={pid}, START", flush=True)
    field = np.array(field, dtype=state["parent"].CType, copy=True)
    support = np.asarray(support, dtype=bool)
    with set_workers(FFT_THREADS_PER_WORKER):
        for _ in range(INNER_REPEATS):
            for hio_update in state["hio_updates"]:
                for _ in range(HIO_ITERATIONS):
                    candidate = data_projection(field, state["pod"], state["amplitude"])
                    field = hio_update(field, candidate, support)
                for _ in range(ER_ITERATIONS):
                    candidate = data_projection(field, state["pod"], state["amplitude"])
                    field = state["er_update"](field, candidate, support)
    if not np.isfinite(field).all():
        raise ValueError(f"Seed {seed} produced a nonfinite field.")
    elapsed = perf_counter() - begin
    finished = time()
    print(f"Stage {cycle}: seed={seed}, PID={pid}, DONE in {elapsed:.2f} s", flush=True)
    return index, np.asarray(field), pid, elapsed, started, finished


def candidate_support(mode):
    blurred = gaussian_filter(np.abs(mode), sigma=SHRINKWRAP_SIGMA)
    maximum = float(blurred.max())
    if not np.isfinite(blurred).all() or maximum <= 0:
        raise ValueError("Cannot construct support from a zero/nonfinite mode.")
    candidate = blurred >= SHRINKWRAP_THRESHOLD * maximum
    if SHRINKWRAP_CLOSING_ITERATIONS:
        candidate = binary_closing(candidate, iterations=SHRINKWRAP_CLOSING_ITERATIONS)
    if not candidate.any():
        raise ValueError("Shrink-wrap produced an empty support.")
    return np.asarray(candidate, dtype=bool)


def masked_phase(field):
    amplitude = np.abs(field)
    return np.ma.masked_where(amplitude <= 1e-8 * amplitude.max(), np.angle(field))


def save_figure(mode, weights, support, candidate, completed_iterations):
    figure, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)
    images = (np.abs(mode), masked_phase(mode), support)
    titles = ("First SVD mode: RMS amplitude", "First SVD mode: phase",
              "Support used; final candidate in red")
    for axis, image, title, cmap in zip(axes[:3], images, titles, ("viridis", "twilight", "gray")):
        options = {"vmin": -np.pi, "vmax": np.pi} if cmap == "twilight" else {}
        artist = axis.imshow(image, cmap=cmap, **options)
        axis.set_title(title)
        figure.colorbar(artist, ax=axis, shrink=0.8)
    if not candidate.all():
        axes[2].contour(candidate, levels=[0.5], colors="red", linewidths=0.8)
    axes[3].bar(np.arange(1, len(weights) + 1), weights)
    axes[3].set_xticks(np.arange(1, len(weights) + 1))
    axes[3].set_ylim(0, 1.05)
    axes[3].set_title("Full four-member SVD spectrum")
    axes[3].set_xlabel("Mode")
    axes[3].set_ylabel("Weight")
    figure.suptitle(
        f"Seeds={SEEDS}; workers={WORKERS}; {completed_iterations} iterations/member; "
        f"first-mode weight={weights[0]:.6f}"
    )
    try:
        figure.savefig(OUTPUT / "reconstruction.png", dpi=150)
    finally:
        plt.close(figure)


def validate_settings():
    if len(SEEDS) != 4 or len(set(SEEDS)) != 4:
        raise ValueError("Provide four distinct seeds.")
    if any(isinstance(s, (bool, np.bool_)) or not isinstance(s, (int, np.integer)) or s < 0 for s in SEEDS):
        raise ValueError("Seeds must be nonnegative integers.")
    if WORKERS != 4:
        raise ValueError("This configuration uses four workers.")
    counts = (HIO_ITERATIONS, ER_ITERATIONS, INNER_REPEATS, OUTER_CYCLES, FFT_THREADS_PER_WORKER)
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in counts):
        raise ValueError("Iteration, stage and FFT thread counts must be positive integers.")
    if not 1 <= SVD_NUM_MODES <= len(SEEDS):
        raise ValueError("Invalid SVD_NUM_MODES.")
    if not np.isfinite(SHRINKWRAP_SIGMA) or SHRINKWRAP_SIGMA <= 0 or not 0 < SHRINKWRAP_THRESHOLD < 1:
        raise ValueError("Invalid shrink-wrap sigma or threshold.")
    if not isinstance(SHRINKWRAP_CLOSING_ITERATIONS, int) or SHRINKWRAP_CLOSING_ITERATIONS < 0:
        raise ValueError("Closing iterations must be a nonnegative integer.")


def main():
    validate_settings()
    intensity, statistics_mask = load_data()
    parameters = make_parameters(intensity, statistics_mask)
    parent = Ptycho(parameters, level=2)
    try:
        pod = next(iter(parent.pods.values()))
        support = cdi_support(parent, pod).copy()
        initial_support = support.copy()
        if support.shape != pod.object.shape:
            raise ValueError("Support and object shapes do not match.")
        if not np.any(pod.diff[pod.mask] > 0):
            raise ValueError("Background subtraction removed all valid signal.")
        fields = np.empty((len(SEEDS), *support.shape), dtype=parent.CType)
        for index, seed in enumerate(SEEDS):
            fields[index] = np.random.default_rng(seed).random(size=support.shape) * support
        initial_fields = fields.copy()
        wp = worker_parameters(parent)
        iterations = INNER_REPEATS * len(BETAS) * (HIO_ITERATIONS + ER_ITERATIONS)
        weights_history, timing_history, support_history = [], [], []
        pid_history, member_seconds_history, intervals_history = [], [], []
        print(f"\nFour-member ensemble; seeds={SEEDS}; workers={WORKERS}", flush=True)
        print(f"Betas={BETAS}; {iterations} iterations/member/stage; stages={OUTER_CYCLES}", flush=True)
        print("No residuals or residual-based selection; all four fields enter SVD.", flush=True)
        print("Main reconstruction = first SVD mode. Members retain their own fields.", flush=True)
        print(f"Phase reset={RESET_PHASE}; FFT threads/worker={FFT_THREADS_PER_WORKER}", flush=True)
        total_start = perf_counter()
        with ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context("spawn"),
                                 initializer=initialize_worker, initargs=(wp,)) as pool:
            for cycle in range(1, OUTER_CYCLES + 1):
                used = support.copy()
                stage_start = perf_counter()
                futures = [pool.submit(run_member, index, seed, fields[index].copy(), used, cycle)
                           for index, seed in enumerate(SEEDS)]
                updated = np.empty_like(fields)
                pids = np.empty(len(SEEDS), dtype=np.int64)
                seconds = np.empty(len(SEEDS))
                intervals = np.empty((len(SEEDS), 2))
                for future in as_completed(futures):
                    index, field, pid, elapsed, started, finished = future.result()
                    updated[index] = field
                    pids[index] = pid
                    seconds[index] = elapsed
                    intervals[index] = (started, finished)
                reconstruction_seconds = perf_counter() - stage_start
                fields = updated
                svd_start = perf_counter()
                modes, weights = svd_modes(fields, num_modes=SVD_NUM_MODES)
                svd_seconds = perf_counter() - svd_start
                shrink_start = perf_counter()
                candidate = candidate_support(modes[0])
                shrink_seconds = perf_counter() - shrink_start
                weights_history.append(weights.copy())
                timing_history.append((reconstruction_seconds, svd_seconds, shrink_seconds))
                support_history.append((used.mean(), candidate.mean()))
                pid_history.append(pids)
                member_seconds_history.append(seconds)
                intervals_history.append(intervals)
                print(f"\nStage {cycle}/{OUTER_CYCLES}: reconstruction={reconstruction_seconds:.2f} s; "
                      f"SVD={svd_seconds:.2f} s; shrink-wrap={shrink_seconds:.2f} s", flush=True)
                print(f"Worker PIDs in seed order: {pids}", flush=True)
                print(f"Member computation seconds: {seconds}", flush=True)
                print(f"SVD weights: {weights}", flush=True)
                print(f"Support fractions: used={used.mean():.6f}, candidate={candidate.mean():.6f}", flush=True)
                if cycle < OUTER_CYCLES:
                    support = candidate.copy()
                    if RESET_PHASE:
                        fields = np.abs(fields).astype(parent.CType)
                    print("Common support applied for the next stage.", flush=True)
                else:
                    print("Final candidate saved but not applied.", flush=True)
        computation_seconds = perf_counter() - total_start
        OUTPUT.mkdir(parents=True, exist_ok=True)
        save_start = perf_counter()
        np.savez_compressed(
            OUTPUT / "reconstruction.npz",
            object_field=modes[0], fields=fields, modes=modes, weights=weights,
            seeds=np.asarray(SEEDS), initial_fields=initial_fields,
            support=used, support_used=used, candidate_support=candidate,
            initial_support=initial_support,
            processed_intensity=pod.diff.copy(), detector_mask=pod.mask.copy(),
            weight_history=np.asarray(weights_history),
            timing_history=np.asarray(timing_history),
            support_fraction_history=np.asarray(support_history),
            worker_pid_history=np.asarray(pid_history),
            member_seconds_history=np.asarray(member_seconds_history),
            worker_time_intervals=np.asarray(intervals_history),
            iterations_per_member=iterations * OUTER_CYCLES,
            completed_cycles=OUTER_CYCLES, workers=WORKERS,
            betas=np.asarray(BETAS), reset_phase=RESET_PHASE,
        )
        save_figure(modes[0], weights, used, candidate, iterations * OUTER_CYCLES)
        print(f"\nComputation including pool lifecycle: {computation_seconds:.2f} s", flush=True)
        print(f"Final saving and plotting: {perf_counter() - save_start:.2f} s", flush=True)
        print(f"Results saved to: {OUTPUT.resolve()}", flush=True)
    finally:
        parent.finalize()


if __name__ == "__main__":
    mp.freeze_support()
    main()
