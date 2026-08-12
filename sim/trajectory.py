"""
Analytic flight trajectories with angular rate as a controlled variable.

The point of the simulator is to make peak angular rate an independent variable
that can be swept, which is impossible with recorded flight. Trajectories are
therefore defined by their angular VELOCITY profile, which is integrated on
SO(3) to obtain orientation; position is analytic.

Because omega(t) is specified rather than differentiated from a pose sequence,
the ground-truth angular velocity is exact -- there is no numerical
differentiation noise anywhere in the pipeline.
"""
import numpy as np


def rotation_matrix_from_rotvec(w):
    """Compute 3x3 rotation matrix from a rotation vector using Rodrigues' formula."""
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


class Trajectory:
    """
    Forward flight down a corridor with oscillatory lateral motion and an
    oscillatory angular-rate profile.

    Parameters
    ----------
    peak_omega : peak angular rate in rad/s -- the swept variable
    omega_hz   : how fast the rotation oscillates
    speed      : forward speed in m/s
    axis       : rotation axis in body frame; default is yaw-dominant, mixed
                 with a little pitch so the motion is not degenerate
    """

    def __init__(self, peak_omega=2.0, omega_hz=0.6, speed=5.0,
                 lateral_amp=0.6, lateral_hz=0.35, vertical_amp=0.25,
                 axis=(0.0, 1.0, 0.15), start_z=1.5, rate_hz=1000.0,
                 broadband_rms=0.0, broadband_seed=7, duration_hint=20.0):
        """
        broadband_rms adds a realistic high-frequency angular-rate component on
        top of the smooth manoeuvre. Measured on the real racing drone, the
        angular rate carries ~0.47 rad/s RMS above 5 Hz (attitude loop plus
        airframe vibration). That content is invisible to a bandwidth-limited
        ground-truth system, so it must be present for any experiment about
        ground-truth bandwidth to be meaningful.
        """
        self.peak_omega = peak_omega
        self.omega_hz = omega_hz
        self.speed = speed
        self.lateral_amp = lateral_amp
        self.lateral_hz = lateral_hz
        self.vertical_amp = vertical_amp
        self.start_z = start_z
        self.rate_hz = rate_hz
        self.broadband_rms = broadband_rms

        axis = np.asarray(axis, dtype=float)
        self.axis = axis / np.linalg.norm(axis)
        self._R_cache = None
        self._t_cache = None
        self._bb_t = None
        if broadband_rms > 0:
            self._build_broadband(duration_hint, broadband_seed)

    def _build_broadband(self, duration, seed):
        """
        Pre-generate a high-frequency angular-rate realisation.

        The spectrum is shaped as 1/f above a 5 Hz corner, which approximates
        the measured falloff on the real vehicle, then scaled so the total RMS
        matches `broadband_rms`.
        """
        n = int(duration * self.rate_hz) + 4
        rng = np.random.default_rng(seed)
        white = rng.standard_normal((n, 3))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.rate_hz)
        shape = np.zeros_like(freqs)
        band = freqs >= 5.0
        shape[band] = 1.0 / freqs[band]
        out = np.zeros((n, 3))
        for j in range(3):
            spec = np.fft.rfft(white[:, j]) * shape
            out[:, j] = np.fft.irfft(spec, n=n)
        rms = np.sqrt((out ** 2).sum(axis=1).mean())
        if rms > 0:
            out *= self.broadband_rms / rms
        self._bb_t = np.arange(n) / self.rate_hz
        self._bb = out

    # -- kinematics -------------------------------------------------------
    def omega(self, t):
        """True body angular velocity (rad/s)."""
        t = np.atleast_1d(t)
        scale = np.sin(2 * np.pi * self.omega_hz * t)
        w = self.peak_omega * scale[:, None] * self.axis[None, :]
        if self._bb_t is not None:
            for j in range(3):
                w[:, j] += np.interp(t, self._bb_t, self._bb[:, j])
        return w

    def position(self, t):
        """True world position (m)."""
        t = np.atleast_1d(t)
        return np.stack([
            self.lateral_amp * np.sin(2 * np.pi * self.lateral_hz * t),
            self.vertical_amp * np.sin(2 * np.pi * self.lateral_hz * 0.7 * t),
            self.start_z + self.speed * t,
        ], axis=1)

    def velocity(self, t, eps=1e-5):
        t = np.atleast_1d(t)
        return (self.position(t + eps) - self.position(t - eps)) / (2 * eps)

    def acceleration(self, t, eps=1e-4):
        t = np.atleast_1d(t)
        return (self.position(t + eps) - 2 * self.position(t)
                + self.position(t - eps)) / (eps ** 2)

    # -- orientation ------------------------------------------------------
    def _build_rotations(self, duration):
        """Integrate omega on SO(3) at `rate_hz` once, then interpolate."""
        n = int(duration * self.rate_hz) + 2
        t = np.arange(n) / self.rate_hz
        w = self.omega(t)
        R = np.zeros((n, 3, 3))
        R[0] = np.eye(3)
        for i in range(1, n):
            dt = t[i] - t[i - 1]
            w_mid = 0.5 * (w[i] + w[i - 1])
            R[i] = R[i - 1] @ rotation_matrix_from_rotvec(w_mid * dt)
        self._t_cache, self._R_cache = t, R

    def rotation(self, t, duration=None):
        """
        World-from-camera rotation at time t.

        Interpolation between the 1 kHz integration samples is done by
        composing the partial step, so it stays on the manifold.
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if self._R_cache is None:
            self._build_rotations(duration if duration else float(t.max()) + 0.1)
        if t.max() > self._t_cache[-1]:
            self._build_rotations(float(t.max()) + 0.1)

        out = np.zeros((len(t), 3, 3))
        for k, tk in enumerate(t):
            i = int(np.clip(np.searchsorted(self._t_cache, tk) - 1,
                            0, len(self._t_cache) - 1))
            dt = tk - self._t_cache[i]
            if dt <= 0:
                out[k] = self._R_cache[i]
            else:
                w_mid = self.omega(np.array([self._t_cache[i]]))[0]
                out[k] = self._R_cache[i] @ rotation_matrix_from_rotvec(w_mid * dt)
        return out

    def pose(self, t, duration=None):
        """(R_wc, t_wc) camera-in-world pose."""
        return self.rotation(t, duration), self.position(t)


class ImuModel:
    """
    Gyroscope and accelerometer with white noise plus bias random walk.

    Defaults are the InvenSense MPU-6150 figures used by the real DAVIS 346
    rig, so simulated inertial performance is comparable to the recorded data.
    """

    def __init__(self, rate_hz=1000.0,
                 gyro_noise_density=0.005,       # rad/s/sqrt(Hz)
                 gyro_bias_walk=4.0e-5,          # rad/s^2/sqrt(Hz)
                 accel_noise_density=0.02,       # m/s^2/sqrt(Hz)
                 accel_bias_walk=4.0e-4,
                 seed=0):
        self.rate_hz = rate_hz
        self.gyro_noise_density = gyro_noise_density
        self.gyro_bias_walk = gyro_bias_walk
        self.accel_noise_density = accel_noise_density
        self.accel_bias_walk = accel_bias_walk
        self.rng = np.random.default_rng(seed)

    def sample(self, traj, duration, gravity=np.array([0.0, 9.81, 0.0])):
        """
        Simulate an IMU stream over [0, duration].

        Returns dict with t, gyro, accel (measured, corrupted) and the true
        values plus the realised bias walks, so estimators can be scored
        against exactly what was injected.
        """
        n = int(duration * self.rate_hz)
        t = np.arange(n) / self.rate_hz
        dt = 1.0 / self.rate_hz

        w_true = traj.omega(t)
        a_world = traj.acceleration(t)
        R = traj.rotation(t, duration)

        # Specific force in body frame: a_body = R^T (a_world + g)
        a_true = np.einsum('nij,nj->ni', np.transpose(R, (0, 2, 1)),
                           a_world + gravity[None, :])

        sigma_g = self.gyro_noise_density * np.sqrt(self.rate_hz)
        sigma_a = self.accel_noise_density * np.sqrt(self.rate_hz)

        bias_g = np.cumsum(
            self.rng.standard_normal((n, 3)) * self.gyro_bias_walk * np.sqrt(dt), axis=0)
        bias_a = np.cumsum(
            self.rng.standard_normal((n, 3)) * self.accel_bias_walk * np.sqrt(dt), axis=0)

        gyro = w_true + bias_g + self.rng.standard_normal((n, 3)) * sigma_g
        accel = a_true + bias_a + self.rng.standard_normal((n, 3)) * sigma_a

        return {'t': t, 'gyro': gyro, 'accel': accel,
                'gyro_true': w_true, 'accel_true': a_true,
                'gyro_bias': bias_g, 'accel_bias': bias_a}
