"""
Piecewise-planar scene with exact analytic depth.

Why planar patches rather than a mesh or a game engine:

  * A textured plane projects into the camera through an exact HOMOGRAPHY, so
    rendering is a single warp per plane -- fast, and free of rasterisation
    approximations.
  * Depth at every pixel is available in CLOSED FORM by intersecting the pixel
    ray with the plane. There is no depth buffer quantisation and no rendering
    noise, so ground-truth depth is exact to floating point.
  * Real texture images can be pasted onto the planes, so feature detectors see
    realistic image statistics rather than synthetic patterns.

Exact depth is the whole point: it is the ground truth that was missing from the
real dataset and that blocked every depth/translation experiment.

Conventions
-----------
World and camera are right-handed. The camera looks down +Z, x right, y down
(standard OpenCV). A camera pose is (R_wc, t_wc): the rotation and position of
the camera IN the world, so a world point maps to camera coordinates by
    X_cam = R_wc^T (X_world - t_wc)
"""
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Plane:
    """
    A finite textured plane.

    origin spans the patch together with `u_axis` and `v_axis`: a texture
    coordinate (s, t) in [0,1]^2 maps to the world point
        origin + s * u_axis + t * v_axis
    so |u_axis| and |v_axis| are the patch's physical extents in metres.
    """
    origin: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    texture: np.ndarray
    name: str = ''

    @property
    def normal(self):
        n = np.cross(self.u_axis, self.v_axis)
        return n / np.linalg.norm(n)

    def corners(self):
        return np.array([
            self.origin,
            self.origin + self.u_axis,
            self.origin + self.u_axis + self.v_axis,
            self.origin + self.v_axis,
        ])


@dataclass
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    read_noise_std: float = 0.0
    distortion: str = 'pinhole'
    D: np.ndarray = None

    @property
    def K(self):
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]])

    @classmethod
    def davis346(cls, noise_std=0.0, use_fisheye=False):
        """Match the real DAVIS 346 used in UZH-FPV, so results transfer."""
        return cls(fx=172.98, fy=172.98, cx=163.34, cy=134.99,
                   width=346, height=260,
                   read_noise_std=noise_std,
                   distortion='equidistant' if use_fisheye else 'pinhole',
                   D=np.array([-0.0039, 0.0469, -0.0456, 0.0135]) if use_fisheye else None)


@dataclass
class Scene:
    planes: list = field(default_factory=list)

    def add(self, plane):
        self.planes.append(plane)
        return self


def _plane_homography(plane, K, R_cw, t_cw):
    """
    Homography mapping texture PIXEL coordinates to image pixel coordinates.

    For a world point X = O + s*U + t*V with (s,t) in [0,1]^2, the camera maps
        x ~ K (R_cw X + t_cw) = K [R_cw U | R_cw V | R_cw O + t_cw] (s, t, 1)^T
    Texture pixels (px, py) relate to (s, t) by s = px / Wt, t = py / Ht.
    """
    Ht_img, Wt_img = plane.texture.shape[:2]
    M = np.column_stack([R_cw @ plane.u_axis,
                         R_cw @ plane.v_axis,
                         R_cw @ plane.origin + t_cw])
    H = K @ M @ np.diag([1.0 / Wt_img, 1.0 / Ht_img, 1.0])
    return H


def _plane_depth_map(plane, cam, R_cw, t_cw):
    """
    Exact per-pixel depth (camera Z) of the plane, analytically.

    A pixel ray in camera coordinates is d = K^-1 [x, y, 1]^T with d_z = 1, so
    the intersection parameter equals the depth directly:
        depth = (n_c . p_c) / (n_c . d)
    where n_c is the plane normal and p_c a plane point, both in camera frame.
    """
    ys, xs = np.mgrid[0:cam.height, 0:cam.width]
    ones = np.ones_like(xs, dtype=np.float64)
    pix = np.stack([xs, ys, ones], axis=-1).astype(np.float64)
    d = pix @ np.linalg.inv(cam.K).T                    # d_z == 1 by construction

    n_c = R_cw @ plane.normal
    p_c = R_cw @ plane.origin + t_cw
    denom = d @ n_c
    numer = float(n_c @ p_c)

    with np.errstate(divide='ignore', invalid='ignore'):
        depth = numer / denom
    depth[~np.isfinite(depth)] = np.inf
    depth[depth <= 0] = np.inf                          # behind the camera
    return depth


def _get_fisheye_map(cam):
    if hasattr(cam, '_fisheye_map'):
        return cam._fisheye_map
    # Virtual wide pinhole camera to cover the fisheye FOV (~115 deg)
    w_wide, h_wide = 800, 800
    f_wide = 200.0
    cam_wide = Camera(fx=f_wide, fy=f_wide, cx=w_wide/2, cy=h_wide/2, width=w_wide, height=h_wide)

    ys, xs = np.mgrid[0:cam.height, 0:cam.width]
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32).reshape(-1, 1, 2)
    rays = cv2.fisheye.undistortPoints(pts, cam.K, cam.D)
    rays = rays.reshape(-1, 2)
    u_wide = rays[:, 0] * f_wide + w_wide/2
    v_wide = rays[:, 1] * f_wide + h_wide/2
    map_x = u_wide.reshape(cam.height, cam.width).astype(np.float32)
    map_y = v_wide.reshape(cam.height, cam.width).astype(np.float32)
    cam._fisheye_map = (cam_wide, map_x, map_y)
    return cam._fisheye_map


def render(scene, cam, R_wc, t_wc, background=0):
    """
    Render the scene from a camera pose.

    Returns (image uint8 HxW, depth float64 HxW). Depth is +inf where no
    surface is visible. Occlusion is resolved by nearest-depth compositing,
    which is exact for planar patches.
    """
    render_cam = cam
    if cam.distortion == 'equidistant':
        render_cam, map_x, map_y = _get_fisheye_map(cam)

    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc

    image = np.full((render_cam.height, render_cam.width), background, dtype=np.uint8)
    depth = np.full((render_cam.height, render_cam.width), np.inf, dtype=np.float64)

    for plane in scene.planes:
        # Skip planes entirely behind the camera.
        corners_cam = (plane.corners() - t_wc) @ R_cw.T
        if np.all(corners_cam[:, 2] <= 1e-6):
            continue

        H = _plane_homography(plane, render_cam.K, R_cw, t_cw)
        if abs(np.linalg.det(H)) < 1e-12:
            continue

        warped = cv2.warpPerspective(
            plane.texture, H, (render_cam.width, render_cam.height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Valid region: where the texture actually lands (warp a white mask).
        mask = cv2.warpPerspective(
            np.full(plane.texture.shape[:2], 255, np.uint8), H,
            (render_cam.width, render_cam.height), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0

        plane_depth = _plane_depth_map(plane, render_cam, R_cw, t_cw)
        visible = mask & (plane_depth < depth)
        depth[visible] = plane_depth[visible]
        image[visible] = warped[visible]

    if cam.distortion == 'equidistant':
        image = cv2.remap(image, map_x, map_y,
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=background)
        depth = cv2.remap(depth.astype(np.float32), map_x, map_y,
                          interpolation=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=np.inf)
        depth = depth.astype(np.float64)

    return image, depth
