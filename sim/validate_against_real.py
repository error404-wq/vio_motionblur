"""
Validate the simulator against the real UZH-FPV measurements.

A simulator is only worth using if it reproduces the behaviour of the system it
stands in for. We therefore configure the simulator to match the real
development clip (Seq 7 [35 s]: mean angular rate 2.63 rad/s, ~5 m/s forward)
and check that it independently reproduces the laws measured on real flight:

  REAL (Seq 7 [35 s], APS)             SIM must reproduce
  ------------------------------------ ---------------------------------
  gyro error 0.72 deg @ 33 ms          calibrated: 0.043 rad/s/sqrt(Hz)
  gyro error ~ H^0.47                  angle random walk
  visual rel. error ~0.44, flat in H    error proportional to rotation
  correspondence 181 -> 43 over 818 ms  attrition with horizon
  APS sharpness drop ~65% at high rate  blur integration

Matching these is what licenses using the simulator to study regimes the real
data cannot reach. Divergences are reported rather than tuned away.
"""
import sys

import numpy as np

sys.path.insert(0, 'vio')

from horizon_analysis import GyroIntegrator, angular_error_deg          # noqa: E402
from visual_horizon import ChainedTracker, rotation_from_correspondences  # noqa: E402

from .scene import Camera                                              # noqa: E402
from .sensors import render_blurred                                    # noqa: E402
from .trajectory import Trajectory, ImuModel                           # noqa: E402
from .worlds import corridor, add_obstacles                            # noqa: E402

# Measured on the real development clip; see PAPER_DRAFT.md R1/R2.
REAL = {
    1:  {'theta': 3.8,  'vis': 1.65, 'gyro': 0.53, 'N': 176},
    3:  {'theta': 10.0, 'vis': 4.50, 'gyro': 1.43, 'N': 155},
    8:  {'theta': 26.6, 'vis': 11.71, 'gyro': 2.51, 'N': 114},
    15: {'theta': 49.8, 'vis': 25.59, 'gyro': 3.21, 'N': 75},
    30: {'theta': 97.3, 'vis': 62.96, 'gyro': 3.55, 'N': 43},
}
HORIZONS = sorted(REAL)


def run(duration=6.0, peak_omega=4.1, speed=5.0, frame_hz=30.0,
        exposure=0.010, blur_samples=6, seed=0, verbose=True):
    """
    Peak omega is set above the real clip's MEAN rate because the trajectory is
    sinusoidal: a sinusoid of peak A has mean |A sin| = 2A/pi, so peak ~4.1
    reproduces a mean of ~2.6 rad/s.
    """
    cam = Camera.davis346()
    scene = add_obstacles(corridor(seed=seed), count=8, seed=100 + seed)
    traj = Trajectory(peak_omega=peak_omega, omega_hz=0.6, speed=speed)

    imu = ImuModel(gyro_noise_density=0.043, seed=seed + 1).sample(traj, duration)
    gyro = GyroIntegrator(imu['t'], imu['gyro'])

    frame_t = np.arange(0.0, duration, 1.0 / frame_hz)
    tracker = ChainedTracker()
    if verbose:
        print(f'rendering {len(frame_t)} frames...', flush=True)
    for i, t in enumerate(frame_t):
        img, _ = render_blurred(scene, cam, traj, t, exposure,
                                blur_samples, duration)
        tracker.process(img)
        if verbose and i % 30 == 0:
            print(f'  {i}/{len(frame_t)}', flush=True)

    R_true = traj.rotation(frame_t, duration)
    mean_omega = float(np.mean(np.linalg.norm(traj.omega(frame_t), axis=1)))

    rows = []
    for k in HORIZONS:
        theta_l, vis_l, gyro_l, n_l = [], [], [], []
        for i in range(len(frame_t) - k):
            t0, t1 = frame_t[i], frame_t[i + k]
            R_gt = R_true[i].T @ R_true[i + k]
            theta = np.degrees(np.arccos(
                np.clip((np.trace(R_gt) - 1) / 2, -1, 1)))
            if theta < 2.0:
                continue
            a, b = tracker.positions[i], tracker.positions[i + k]
            shared = sorted(set(a) & set(b))
            n_l.append(len(shared))
            if len(shared) < 8:
                continue
            p0 = np.array([a[j] for j in shared], dtype=np.float32)
            p1 = np.array([b[j] for j in shared], dtype=np.float32)
            est = rotation_from_correspondences(p0, p1)
            if est is None:
                continue
            theta_l.append(theta)
            vis_l.append(angular_error_deg(est[0], R_gt))
            gyro_l.append(angular_error_deg(gyro.R_rel(t0, t1), R_gt))
        if theta_l:
            rows.append({'k': k,
                         'theta': float(np.median(theta_l)),
                         'vis': float(np.median(vis_l)),
                         'gyro': float(np.median(gyro_l)),
                         'N': float(np.median(n_l))})
    return rows, mean_omega


def report(rows, mean_omega):
    print()
    print(f'SIMULATOR vs REAL   (sim mean omega {mean_omega:.2f} rad/s, real 2.63)')
    print(f'{"horizon":>8s} | {"theta sim/real":>16s} | {"vis sim/real":>15s} '
          f'| {"gyro sim/real":>15s} | {"N sim/real":>13s}')
    print('-' * 82)
    rel_sim, rel_real = [], []
    for r in rows:
        ref = REAL[r['k']]
        print(f'{r["k"] * 33:6.0f}ms | {r["theta"]:7.1f} /{ref["theta"]:7.1f} '
              f'| {r["vis"]:6.2f} /{ref["vis"]:6.2f} '
              f'| {r["gyro"]:6.2f} /{ref["gyro"]:6.2f} '
              f'| {r["N"]:5.0f} /{ref["N"]:5.0f}')
        rel_sim.append(r['vis'] / r['theta'])
        rel_real.append(ref['vis'] / ref['theta'])
    print('-' * 82)
    print(f'visual RELATIVE error  sim {np.mean(rel_sim):.3f}  vs  real {np.mean(rel_real):.3f}')
    print('(real law: relative error flat in horizon at ~0.44 -- error tracks rotation)')


if __name__ == '__main__':
    rows, mean_omega = run()
    report(rows, mean_omega)
