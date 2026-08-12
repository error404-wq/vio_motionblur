"""
GROUND-TRUTH BANDWIDTH AS AN EVALUATION CONFOUND.

On real UZH-FPV flight, visual and gyroscopic rotation errors correlate at
rho = +0.64, which caps measured fusion gain at ~2.5% instead of the ~23% that
independent errors would give. Spectral analysis showed why: the ground-truth
orientation is low-pass filtered, retaining almost nothing above 15 Hz, while
the vehicle genuinely carries ~0.47 rad/s RMS of angular rate above 5 Hz. Both
sensors observe the true motion and are scored against a smoothed reference, so
the discarded motion becomes a COMMON error injected into both.

This experiment reproduces that mechanism under full control. The simulator has
an exact reference, so we can degrade it deliberately:

  1. evaluate against the EXACT reference        -> errors should be independent
                                                    and fusion should pay in full
  2. evaluate against LOW-PASS references         -> as bandwidth falls, apparent
                                                    errors inflate, correlation
                                                    rises, apparent fusion gain
                                                    collapses

If a cutoff near the real system's reproduces the real numbers, that closes the
argument: the real-world "fusion does not help" conclusion is an artifact of the
benchmark's reference, not a property of the sensors.

The filtered reference is built by low-pass filtering the TRUE angular velocity
and re-integrating it, which is exactly what a filtered tracking system yields.
"""
import sys

import cv2
import numpy as np
from scipy import signal, stats
from scipy.spatial.transform import Rotation

sys.path.insert(0, 'vio')

from horizon_analysis import GyroIntegrator, angular_error_deg     # noqa: E402
from visual_horizon import ChainedTracker                          # noqa: E402

from .scene import Camera                                          # noqa: E402
from .sensors import render_blurred                                # noqa: E402
from .test_error_correlation import rotation_pinhole               # noqa: E402
from .trajectory import Trajectory, ImuModel                       # noqa: E402
from .worlds import corridor, add_obstacles                        # noqa: E402

CUTOFFS = [2.0, 5.0, 10.0, 20.0, 50.0, None]      # None = exact reference
WEIGHTS = np.linspace(0.0, 1.0, 41)


def integrate_omega(t, w):
    """Cumulative SO(3) rotation from an angular-rate series."""
    R = np.zeros((len(t), 3, 3))
    R[0] = np.eye(3)
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        w_mid = 0.5 * (w[i] + w[i - 1])
        R[i] = R[i - 1] @ Rotation.from_rotvec(w_mid * dt).as_matrix()
    return R


def filtered_reference(traj, duration, rate_hz, cutoff):
    """Reference rotation trajectory from low-pass filtered true angular rate."""
    t = np.arange(0.0, duration, 1.0 / rate_hz)
    w = traj.omega(t)
    if cutoff is not None:
        b, a = signal.butter(4, cutoff / (0.5 * rate_hz), btype='low')
        w = signal.filtfilt(b, a, w, axis=0)      # zero-phase: no lag introduced
    return t, integrate_omega(t, w)


def rel_from_series(t_series, R_series, t0, t1):
    i = int(np.clip(np.searchsorted(t_series, t0) - 1, 0, len(t_series) - 1))
    j = int(np.clip(np.searchsorted(t_series, t1) - 1, 0, len(t_series) - 1))
    return R_series[i].T @ R_series[j]


def fuse(R_gyro, R_vis, w):
    rvec = Rotation.from_matrix(R_gyro.T @ R_vis).as_rotvec()
    return R_gyro @ Rotation.from_rotvec(w * rvec).as_matrix()


def run(duration=6.0, peak_omega=4.1, speed=5.0, frame_hz=30.0,
        broadband_rms=0.47, gyro_noise_density=0.005, seed=0, verbose=True):
    """
    Note on gyro noise: the earlier 0.043 figure was calibrated against the
    SMOOTHED real reference, so it absorbed the reference error. Here we use the
    datasheet value, because the reference degradation is modelled explicitly.
    """
    cam = Camera.davis346()
    scene = add_obstacles(corridor(seed=seed), count=8, seed=100 + seed)
    traj = Trajectory(peak_omega=peak_omega, omega_hz=0.6, speed=speed,
                      broadband_rms=broadband_rms, duration_hint=duration + 1)

    imu = ImuModel(gyro_noise_density=gyro_noise_density,
                   seed=seed + 1).sample(traj, duration)
    gyro = GyroIntegrator(imu['t'], imu['gyro'])

    frame_t = np.arange(0.0, duration, 1.0 / frame_hz)
    tracker = ChainedTracker()
    for i, t in enumerate(frame_t):
        img, _ = render_blurred(scene, cam, traj, t, 0.010, 6, duration)
        tracker.process(img)
        if verbose and i % 40 == 0:
            print(f'  rendered {i}/{len(frame_t)}', flush=True)

    # Visual and gyro estimates, computed ONCE; only the reference changes.
    est = []
    for i in range(len(frame_t) - 1):
        a, b = tracker.positions[i], tracker.positions[i + 1]
        shared = sorted(set(a) & set(b))
        if len(shared) < 8:
            continue
        p0 = np.array([a[j] for j in shared], np.float32)
        p1 = np.array([b[j] for j in shared], np.float32)
        R_vis = rotation_pinhole(p0, p1, cam.K)
        if R_vis is None:
            continue
        est.append((frame_t[i], frame_t[i + 1],
                    gyro.R_rel(frame_t[i], frame_t[i + 1]), R_vis))

    rows = []
    for cutoff in CUTOFFS:
        t_ref, R_ref = filtered_reference(traj, duration, 1000.0, cutoff)
        eg, ev = [], []
        Eg, Ev = [], []
        for t0, t1, R_gyro, R_vis in est:
            R_gt = rel_from_series(t_ref, R_ref, t0, t1)
            eg.append(angular_error_deg(R_gyro, R_gt))
            ev.append(angular_error_deg(R_vis, R_gt))
            Eg.append(cv2.Rodrigues(R_gyro @ R_gt.T)[0].ravel())
            Ev.append(cv2.Rodrigues(R_vis @ R_gt.T)[0].ravel())
        eg, ev = np.array(eg), np.array(ev)
        Eg, Ev = np.array(Eg), np.array(Ev)

        rho = float(stats.spearmanr(eg, ev).statistic)
        cos = np.einsum('ij,ij->i', Eg, Ev) / (
            np.linalg.norm(Eg, axis=1) * np.linalg.norm(Ev, axis=1) + 1e-12)
        vec_ang = float(np.degrees(np.arccos(np.clip(np.median(cos), -1, 1))))

        best_w, best_e = 0.0, float(np.median(eg))
        for w in WEIGHTS:
            e = float(np.median([angular_error_deg(fuse(g, v, w),
                                 rel_from_series(t_ref, R_ref, t0, t1))
                                 for t0, t1, g, v in est]))
            if e < best_e:
                best_w, best_e = float(w), e

        rows.append({'cutoff': cutoff, 'gyro': float(np.median(eg)),
                     'vis': float(np.median(ev)), 'rho': rho,
                     'vec_angle': vec_ang, 'best_w': best_w,
                     'fused': best_e,
                     'gain': 100.0 * (1 - best_e / float(np.median(eg)))})
    return rows


def report(rows):
    print()
    print('APPARENT SENSOR PERFORMANCE vs GROUND-TRUTH BANDWIDTH')
    print(f'{"GT cutoff":>11s}{"gyro":>8s}{"vision":>8s}{"rho":>8s}'
          f'{"err-vec":>9s}{"best w":>8s}{"fused":>8s}{"gain":>8s}')
    print('-' * 66)
    for r in rows:
        label = 'exact' if r['cutoff'] is None else f'{r["cutoff"]:.0f} Hz'
        print(f'{label:>11s}{r["gyro"]:8.3f}{r["vis"]:8.3f}{r["rho"]:+8.3f}'
              f'{r["vec_angle"]:8.0f}d{r["best_w"]:8.2f}{r["fused"]:8.3f}'
              f'{r["gain"]:7.1f}%')
    print('-' * 66)
    print('REAL UZH-FPV (GT retains almost nothing above ~15 Hz):')
    print(f'{"real":>11s}{0.530:8.3f}{0.649:8.3f}{0.641:+8.3f}{34:8.0f}d'
          f'{0.20:8.2f}{0.506:8.3f}{4.5:7.1f}%')


if __name__ == '__main__':
    print('Simulating with realistic high-frequency motion content...')
    report(run())
