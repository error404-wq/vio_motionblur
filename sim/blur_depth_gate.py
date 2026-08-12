"""
FEASIBILITY GATE: is motion blur a usable dense depth cue?

Physics
-------
Image motion at a pixel decomposes into a rotational and a translational part:

    flow = flow_rot(omega)  +  (1/Z) * A(pixel) * v

The rotational term is INDEPENDENT of depth. The translational term is
proportional to inverse depth. Over an exposure T the pixel smears along the
flow direction by an amount proportional to |flow| * T, so within a single
blurred frame the blur EXTENT varies with depth while the blur DIRECTION is
fixed by the (IMU-known) motion.

That makes depth recovery a 1-D estimation problem per pixel: the direction is
given, only the length is unknown. And the signal grows with speed and exposure
-- the opposite of every correspondence-based method, which degrades with both.

What this gate tests
--------------------
Whether the signal exists at all, before any estimator is built:

  1. predicted translational blur extent, computed analytically from exact
     depth and exact velocity
  2. measured blur anisotropy: a blurred patch loses gradient energy ALONG the
     blur direction while retaining it perpendicular, so the log-ratio of
     perpendicular to parallel gradient energy is a direct, parameter-free
     proxy for blur length

If (2) tracks (1) across the depth range, the cue is real.
"""
import numpy as np

from .scene import render


def flow_field(cam, depth, R_wc, v_world, omega_body):
    """
    Analytic optical flow (pixels/second) and its rotational-only part.

    Velocities are converted into the camera frame; the standard continuous
    flow equations are then applied in normalised coordinates and scaled by the
    focal length.
    """
    ys, xs = np.mgrid[0:cam.height, 0:cam.width]
    u = (xs - cam.cx) / cam.fx
    w = (ys - cam.cy) / cam.fy

    R_cw = R_wc.T
    v = R_cw @ v_world                    # linear velocity in camera frame
    wx, wy, wz = omega_body               # angular velocity in camera frame

    # Rotational component (depth independent)
    u_rot = -wx * u * w + wy * (1 + u ** 2) - wz * w
    w_rot = -wx * (1 + w ** 2) + wy * u * w + wz * u

    # Translational component (scales with inverse depth)
    inv_z = np.where(np.isfinite(depth), 1.0 / np.maximum(depth, 1e-6), 0.0)
    u_tr = inv_z * (-v[0] + u * v[2])
    w_tr = inv_z * (-v[1] + w * v[2])

    fu_rot, fw_rot = u_rot * cam.fx, w_rot * cam.fy
    fu_tr, fw_tr = u_tr * cam.fx, w_tr * cam.fy
    return (fu_rot + fu_tr, fw_rot + fw_tr), (fu_rot, fw_rot), (fu_tr, fw_tr)


def directional_gradient_energy(image, dir_x, dir_y, patch=9):
    """
    Log-ratio of gradient energy perpendicular to versus along a given
    direction, computed over local patches.

    Blur suppresses image detail ALONG the smear direction and leaves detail
    perpendicular to it untouched, so this ratio increases with blur length. It
    requires no knowledge of the blur kernel and has no tuned parameters.
    """
    img = image.astype(np.float64) / 255.0
    gy, gx = np.gradient(img)

    norm = np.sqrt(dir_x ** 2 + dir_y ** 2) + 1e-9
    ux, uy = dir_x / norm, dir_y / norm

    g_par = gx * ux + gy * uy              # along the smear
    g_perp = -gx * uy + gy * ux            # across the smear

    kernel = np.ones((patch, patch)) / (patch * patch)
    from scipy.signal import fftconvolve
    e_par = fftconvolve(g_par ** 2, kernel, mode='same')
    e_perp = fftconvolve(g_perp ** 2, kernel, mode='same')
    return 0.5 * np.log((e_perp + 1e-8) / (e_par + 1e-8))


def run_gate(scene, cam, traj, t_center, exposure=0.020, n_samples=16,
             duration=None):
    """Render one blurred frame and compare measured anisotropy to prediction."""
    from .sensors import render_blurred

    blurred, depth = render_blurred(scene, cam, traj, t_center, exposure,
                                    n_samples, duration)
    sharp, _ = render(scene, cam,
                      traj.rotation(np.array([t_center]), duration)[0],
                      traj.position(np.array([t_center]))[0])

    R_wc = traj.rotation(np.array([t_center]), duration)[0]
    v_world = traj.velocity(np.array([t_center]))[0]
    omega = traj.omega(np.array([t_center]))[0]

    total, rot, trans = flow_field(cam, depth, R_wc, v_world, omega)
    blur_total = np.hypot(total[0], total[1]) * exposure
    blur_trans = np.hypot(trans[0], trans[1]) * exposure

    aniso_blur = directional_gradient_energy(blurred, total[0], total[1])
    aniso_sharp = directional_gradient_energy(sharp, total[0], total[1])
    return {
        'blurred': blurred, 'sharp': sharp, 'depth': depth,
        'blur_total_px': blur_total, 'blur_trans_px': blur_trans,
        'aniso_blur': aniso_blur, 'aniso_sharp': aniso_sharp,
        'inv_depth': np.where(np.isfinite(depth), 1.0 / np.maximum(depth, 1e-6), 0.0),
    }
