"""
Camera sensor models: motion blur and event generation.

Both are simulated by rendering the scene at a rate far above the frame rate
and then applying the appropriate physical model, rather than by post-hoc image
filtering. That matters: motion blur under fast rotation is not a convolution
with a fixed kernel -- the blur trajectory is different for every pixel because
the projective flow is spatially varying. Integrating true sub-exposures
reproduces this exactly.

Event generation follows the standard log-intensity threshold model used by
ESIM and v2e: a pixel emits an event whenever its log intensity has moved by
more than a contrast threshold C since that pixel's last event.
"""
import numpy as np

from .scene import render

LOG_EPS = 1e-3          # guards log(0) for black pixels


def _log_intensity(image):
    return np.log(image.astype(np.float64) / 255.0 + LOG_EPS)


def render_blurred(scene, cam, traj, t_center, exposure=0.010, n_samples=8,
                   duration=None):
    """
    Render one frame with physically-integrated motion blur.

    The shutter is open over [t_center - exposure/2, t_center + exposure/2];
    the frame is the mean of `n_samples` instantaneous renders across it. Depth
    is reported at the exposure midpoint.
    """
    times = np.linspace(t_center - exposure / 2.0,
                        t_center + exposure / 2.0, n_samples)
    R = traj.rotation(times, duration)
    p = traj.position(times)

    accum = np.zeros((cam.height, cam.width), dtype=np.float64)
    for k in range(n_samples):
        img, _ = render(scene, cam, R[k], p[k])
        accum += img

    blurred = accum / n_samples
    if hasattr(cam, 'read_noise_std') and cam.read_noise_std > 0:
        noise = np.random.normal(0, cam.read_noise_std, blurred.shape)
        blurred = np.clip(blurred + noise, 0, 255)

    blurred = blurred.astype(np.uint8)

    R_mid = traj.rotation(np.array([t_center]), duration)[0]
    p_mid = traj.position(np.array([t_center]))[0]
    _, depth = render(scene, cam, R_mid, p_mid)
    return blurred, depth


class EventCamera:
    """
    Log-intensity threshold event model.

    Parameters
    ----------
    contrast_threshold : log-intensity change required to emit an event
    refractory_s       : minimum time between events at the same pixel, which
                         is what bounds the per-pixel rate in real sensors
    """

    def __init__(self, contrast_threshold=0.20, refractory_s=1e-4, seed=0,
                 threshold_sigma=0.03):
        self.C = contrast_threshold
        self.refractory_s = refractory_s
        self.rng = np.random.default_rng(seed)
        self.threshold_sigma = threshold_sigma
        self._ref = None            # per-pixel reference log intensity
        self._last_t = None         # per-pixel time of last event

    def reset(self, image, t0):
        self._ref = _log_intensity(image)
        self._last_t = np.full(image.shape, -np.inf)

    def update(self, image, t):
        """
        Emit events explaining the change since the last call.

        Returns (x, y, t, polarity) arrays. Event timestamps are assigned at
        the sample time; sampling faster than the motion makes this a good
        approximation to the true continuous-time event times.
        """
        log_i = _log_intensity(image)
        if self._ref is None:
            self.reset(image, t)
            return (np.empty(0, int), np.empty(0, int),
                    np.empty(0), np.empty(0, int))

        diff = log_i - self._ref
        # Per-pixel threshold jitter, as in real sensors.
        thresh = self.C * (1.0 + self.threshold_sigma *
                           self.rng.standard_normal(diff.shape))
        n_events = np.floor(np.abs(diff) / np.maximum(thresh, 1e-6))
        fired = (n_events >= 1) & ((t - self._last_t) > self.refractory_s)

        ys, xs = np.nonzero(fired)
        if len(xs) == 0:
            return (np.empty(0, int), np.empty(0, int),
                    np.empty(0), np.empty(0, int))

        pol = np.sign(diff[ys, xs]).astype(int)
        # Advance the reference by the number of thresholds actually crossed.
        self._ref[ys, xs] += pol * n_events[ys, xs] * thresh[ys, xs]
        self._last_t[ys, xs] = t
        return xs, ys, np.full(len(xs), t), pol


def simulate_stream(scene, cam, traj, duration, frame_hz=30.0,
                    event_render_hz=2000.0, exposure=0.010, blur_samples=8,
                    contrast_threshold=0.20, seed=0, verbose=False):
    """
    Produce a synchronised APS + event + depth stream for one flight.

    The event stream is generated from renders at `event_render_hz`, which must
    substantially exceed the image motion bandwidth for the threshold model to
    be accurate.
    """
    ev_dt = 1.0 / event_render_hz
    ev_times = np.arange(0.0, duration, ev_dt)
    cam_times = np.arange(0.0, duration, 1.0 / frame_hz)

    evcam = EventCamera(contrast_threshold=contrast_threshold, seed=seed)
    ex, ey, et, ep = [], [], [], []

    for i, t in enumerate(ev_times):
        R = traj.rotation(np.array([t]), duration)[0]
        p = traj.position(np.array([t]))[0]
        img, _ = render(scene, cam, R, p)
        if i == 0:
            evcam.reset(img, t)
            continue
        xs, ys, ts, pol = evcam.update(img, t)
        if len(xs):
            ex.append(xs)
            ey.append(ys)
            et.append(ts)
            ep.append(pol)
        if verbose and i % 200 == 0:
            print(f'  events {t:.2f}/{duration:.1f}s', flush=True)

    frames, depths = [], []
    for t in cam_times:
        img, dep = render_blurred(scene, cam, traj, t, exposure,
                                  blur_samples, duration)
        frames.append(img)
        depths.append(dep)

    def cat(a, d):
        return np.concatenate(a) if a else np.empty(0, d)

    return {
        'frame_t': cam_times, 'frames': frames, 'depths': depths,
        'ev_x': cat(ex, int), 'ev_y': cat(ey, int),
        'ev_t': cat(et, float), 'ev_p': cat(ep, int),
    }
