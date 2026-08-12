"""
Browser build of the simulator — NumPy only.

The desktop package uses OpenCV (homography warp) and SciPy (rotations). In
Pyodide those two wheels dominate load time (~50 MB combined), so this module
reimplements exactly what the web demo needs in pure NumPy: Pyodide then only
has to fetch NumPy (~8 MB), which is the difference between a page that loads
and one that does not.

Physics is unchanged. Rendering is by direct ray casting rather than a
homography warp, which is if anything more faithful: it handles the equidistant
fisheye exactly instead of warping a pinhole render, and depth still comes from
a closed-form ray-plane intersection.
"""
import numpy as np

D_FISHEYE = np.array([-0.0039, 0.0469, -0.0456, 0.0135])


# ── rotations (replaces scipy.spatial.transform.Rotation) ────────────────
def rodrigues(rvec):
    """Rotation matrix from a rotation vector."""
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3)
    k = np.asarray(rvec, dtype=float) / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


# ── camera ───────────────────────────────────────────────────────────────
class Camera:
    def __init__(self, fx, fy, cx, cy, width, height,
                 noise_std=0.0, fisheye=False):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.width, self.height = width, height
        self.noise_std = noise_std
        self.fisheye = fisheye
        self._rays = None

    @classmethod
    def davis346(cls, noise_std=0.0, use_fisheye=False, scale=1.0):
        w, h = int(346 * scale), int(260 * scale)
        return cls(172.98 * scale, 172.98 * scale, 163.34 * scale, 134.99 * scale,
                   w, h, noise_std, use_fisheye)

    def rays(self):
        """Unit ray direction per pixel, in camera coordinates. Cached."""
        if self._rays is not None:
            return self._rays
        ys, xs = np.mgrid[0:self.height, 0:self.width]
        x = (xs - self.cx) / self.fx
        y = (ys - self.cy) / self.fy
        if self.fisheye:
            # equidistant: image radius = f * theta_d, theta_d = theta*(1+k.theta^2..)
            r = np.sqrt(x * x + y * y)
            th = r.copy()
            for _ in range(6):                       # invert the polynomial
                t2 = th * th
                f = th * (1 + D_FISHEYE[0] * t2 + D_FISHEYE[1] * t2**2
                          + D_FISHEYE[2] * t2**3 + D_FISHEYE[3] * t2**4) - r
                df = (1 + 3 * D_FISHEYE[0] * t2 + 5 * D_FISHEYE[1] * t2**2
                      + 7 * D_FISHEYE[2] * t2**3 + 9 * D_FISHEYE[3] * t2**4)
                th = th - f / np.maximum(df, 1e-9)
            scale = np.where(r > 1e-9, np.tan(np.clip(th, 0, 1.45)) / np.maximum(r, 1e-9), 1.0)
            x, y = x * scale, y * scale
        d = np.stack([x, y, np.ones_like(x)], axis=-1)
        self._rays = d / np.linalg.norm(d, axis=-1, keepdims=True)
        return self._rays


# ── scene ────────────────────────────────────────────────────────────────
class Plane:
    def __init__(self, origin, u_axis, v_axis, texture, name=''):
        self.origin = np.asarray(origin, float)
        self.u = np.asarray(u_axis, float)
        self.v = np.asarray(v_axis, float)
        self.texture = texture
        self.name = name
        n = np.cross(self.u, self.v)
        self.normal = n / np.linalg.norm(n)
        self.uu = self.u @ self.u
        self.vv = self.v @ self.v


class Scene:
    def __init__(self):
        self.planes = []

    def add(self, plane):
        self.planes.append(plane)
        return self


def natural_texture(size=256, alpha=1.2, seed=0, contrast=1.0):
    """1/f^alpha noise — natural image statistics, so trackers behave realistically."""
    rng = np.random.default_rng(seed)
    spec = np.fft.fftshift(np.fft.fft2(rng.standard_normal((size, size))))
    fy, fx = np.mgrid[-size // 2:size // 2, -size // 2:size // 2]
    rad = np.sqrt(fx ** 2 + fy ** 2)
    rad[size // 2, size // 2] = 1.0
    img = np.real(np.fft.ifft2(np.fft.ifftshift(spec / rad ** alpha)))
    img -= img.mean()
    s = img.std()
    if s > 0:
        img /= s
    return np.clip(128 + 45 * contrast * img, 0, 255).astype(np.uint8)


def corridor(length=30.0, width=6.0, height=3.0, seed=0, tex=256):
    s, half = Scene(), width / 2

    def T(k, a=1.2, c=1.0):
        return natural_texture(tex, a, seed + k, c)

    s.add(Plane([-half, height / 2, 0], [width, 0, 0], [0, 0, length], T(1), 'floor'))
    s.add(Plane([-half, -height / 2, 0], [width, 0, 0], [0, 0, length], T(2, 1.4), 'ceiling'))
    s.add(Plane([-half, -height / 2, 0], [0, height, 0], [0, 0, length], T(3), 'wall_l'))
    s.add(Plane([half, -height / 2, 0], [0, height, 0], [0, 0, length], T(4), 'wall_r'))
    s.add(Plane([-half, -height / 2, length], [width, 0, 0], [0, height, 0], T(5, 1.2, 1.2), 'end'))
    return s


def add_obstacles(scene, count=8, length=30.0, width=6.0, height=3.0, seed=100, tex=128):
    rng = np.random.default_rng(seed)
    for i in range(count):
        z = rng.uniform(3, length - 3)
        x = rng.uniform(-width / 2 + .5, width / 2 - 1.5)
        y = rng.uniform(-height / 2 + .3, height / 2 - 1)
        w, h = rng.uniform(.4, 1.2), rng.uniform(.4, 1.2)
        yaw = rng.uniform(-.5, .5)
        scene.add(Plane([x, y, z], [w * np.cos(yaw), 0, w * np.sin(yaw)], [0, h, 0],
                        natural_texture(tex, 1.1, seed + i, 1.3), f'obs{i}'))
    return scene


# ── rendering ────────────────────────────────────────────────────────────
def render(scene, cam, R_wc, t_wc):
    """Ray-cast the scene. Returns (uint8 image, float depth in metres)."""
    R_cw = R_wc.T
    d_world = cam.rays() @ R_cw            # ray dirs in world frame
    o = np.asarray(t_wc, float)

    H, W = cam.height, cam.width
    depth = np.full((H, W), np.inf)
    image = np.zeros((H, W), np.uint8)

    for p in scene.planes:
        denom = d_world @ p.normal
        ok = np.abs(denom) > 1e-9
        if not ok.any():
            continue
        t = np.where(ok, ((p.origin - o) @ p.normal) / np.where(ok, denom, 1), -1.0)
        hit = ok & (t > 1e-4) & (t < depth)
        if not hit.any():
            continue

        X = o + d_world[hit] * t[hit][:, None]      # world hit points
        rel = X - p.origin
        su = (rel @ p.u) / p.uu
        sv = (rel @ p.v) / p.vv
        inside = (su >= 0) & (su < 1) & (sv >= 0) & (sv < 1)
        if not inside.any():
            continue

        idx = np.nonzero(hit)
        idx = (idx[0][inside], idx[1][inside])
        th, tw = p.texture.shape
        tx = np.clip((su[inside] * tw).astype(int), 0, tw - 1)
        ty = np.clip((sv[inside] * th).astype(int), 0, th - 1)
        image[idx] = p.texture[ty, tx]
        depth[idx] = t[hit][inside]

    if cam.noise_std > 0:
        n = np.random.normal(0, cam.noise_std, image.shape)
        image = np.clip(image.astype(np.float32) + n, 0, 255).astype(np.uint8)
    return image, depth


def render_blurred(scene, cam, traj, t_center, exposure=0.010, n_samples=4, duration=None):
    """Motion blur by integrating true sub-exposures across the shutter."""
    ts = np.linspace(t_center - exposure / 2, t_center + exposure / 2, max(n_samples, 1))
    acc = np.zeros((cam.height, cam.width), np.float64)
    noise, cam.noise_std = cam.noise_std, 0.0        # add noise once, after integrating
    for t in ts:
        img, _ = render(scene, cam, traj.rotation_at(t), traj.position_at(t))
        acc += img
    cam.noise_std = noise
    out = acc / len(ts)
    if noise > 0:
        out = out + np.random.normal(0, noise, out.shape)
    _, depth = render(scene, cam, traj.rotation_at(t_center), traj.position_at(t_center))
    return np.clip(out, 0, 255).astype(np.uint8), depth


# ── trajectory & IMU ─────────────────────────────────────────────────────
class Trajectory:
    def __init__(self, peak_omega=2.0, omega_hz=0.6, speed=5.0,
                 lateral_amp=0.6, lateral_hz=0.35, vertical_amp=0.25,
                 axis=(0.0, 1.0, 0.15), start_z=1.5, rate_hz=400.0):
        self.peak_omega, self.omega_hz, self.speed = peak_omega, omega_hz, speed
        self.lateral_amp, self.lateral_hz = lateral_amp, lateral_hz
        self.vertical_amp, self.start_z = vertical_amp, start_z
        self.rate_hz = rate_hz
        a = np.asarray(axis, float)
        self.axis = a / np.linalg.norm(a)
        self._t = None

    def omega(self, t):
        t = np.atleast_1d(t)
        return self.peak_omega * np.sin(2 * np.pi * self.omega_hz * t)[:, None] * self.axis

    def position(self, t):
        t = np.atleast_1d(t)
        return np.stack([
            self.lateral_amp * np.sin(2 * np.pi * self.lateral_hz * t),
            self.vertical_amp * np.sin(2 * np.pi * self.lateral_hz * 0.7 * t),
            self.start_z + self.speed * t], axis=1)

    def position_at(self, t):
        return self.position(np.array([float(t)]))[0]

    def _build(self, duration):
        n = int(duration * self.rate_hz) + 2
        ts = np.arange(n) / self.rate_hz
        w = self.omega(ts)
        R = np.zeros((n, 3, 3))
        R[0] = np.eye(3)
        for i in range(1, n):
            dt = ts[i] - ts[i - 1]
            R[i] = R[i - 1] @ rodrigues(0.5 * (w[i] + w[i - 1]) * dt)
        self._t, self._R = ts, R

    def rotation_at(self, t):
        t = float(t)
        if self._t is None or t > self._t[-1]:
            self._build(max(t + 0.2, 1.0))
        i = int(np.clip(np.searchsorted(self._t, t) - 1, 0, len(self._t) - 1))
        dt = t - self._t[i]
        return self._R[i] if dt <= 0 else self._R[i] @ rodrigues(self.omega(self._t[i])[0] * dt)


class ImuModel:
    def __init__(self, rate_hz=400.0, gyro_noise_density=0.043,
                 gyro_bias_walk=4e-5, seed=0):
        self.rate_hz, self.nd, self.bw = rate_hz, gyro_noise_density, gyro_bias_walk
        self.rng = np.random.default_rng(seed)

    def sample(self, traj, duration):
        n = int(duration * self.rate_hz)
        t = np.arange(n) / self.rate_hz
        true = traj.omega(t)
        dt = 1.0 / self.rate_hz
        bias = np.cumsum(self.rng.standard_normal((n, 3)) * self.bw * np.sqrt(dt), axis=0)
        meas = true + bias + self.rng.standard_normal((n, 3)) * self.nd * np.sqrt(self.rate_hz)
        return {'t': t, 'gyro': meas, 'gyro_true': true}
