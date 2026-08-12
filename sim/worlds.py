"""
Scene construction and texture synthesis.

Textures are synthesised with a 1/f^alpha power spectrum, which is the spectral
signature of natural images. This matters: feature detectors and KLT respond to
the local gradient statistics of the image, so a texture with natural spectral
falloff produces realistic corner densities and tracking behaviour, whereas
white noise or checkerboards do not.
"""
import numpy as np

from .scene import Plane, Scene


def natural_texture(size=512, alpha=1.2, seed=0, contrast=1.0):
    """
    Grayscale texture with a 1/f^alpha amplitude spectrum.

    alpha ~ 1.0-1.5 matches natural scenes. Larger alpha gives smoother,
    lower-frequency texture (fewer trackable corners).
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((size, size))
    spectrum = np.fft.fftshift(np.fft.fft2(noise))

    fy, fx = np.mgrid[-size // 2:size // 2, -size // 2:size // 2]
    radius = np.sqrt(fx ** 2 + fy ** 2)
    radius[size // 2, size // 2] = 1.0
    spectrum /= radius ** alpha

    img = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum)))
    img -= img.mean()
    std = img.std()
    if std > 0:
        img /= std
    img = 128.0 + 45.0 * contrast * img
    return np.clip(img, 0, 255).astype(np.uint8)


def corridor(length=30.0, width=6.0, height=3.0, seed=0, texture_size=768):
    """
    A corridor: floor, ceiling, two side walls, and an end wall.

    Deliberately chosen because it produces a wide range of scene depths
    (metres to tens of metres) within a single view, which is the regime where
    depth-dependent effects such as parallax and correspondence survival become
    visible.
    """
    scene = Scene()

    def tex(s, a=1.2, c=1.0):
        return natural_texture(texture_size, alpha=a, seed=s, contrast=c)

    half = width / 2.0

    # Floor and ceiling span the corridor length.
    scene.add(Plane(np.array([-half, height / 2, 0.0]),
                    np.array([width, 0.0, 0.0]),
                    np.array([0.0, 0.0, length]),
                    tex(seed + 1), 'floor'))
    scene.add(Plane(np.array([-half, -height / 2, 0.0]),
                    np.array([width, 0.0, 0.0]),
                    np.array([0.0, 0.0, length]),
                    tex(seed + 2, a=1.4), 'ceiling'))

    # Side walls.
    scene.add(Plane(np.array([-half, -height / 2, 0.0]),
                    np.array([0.0, height, 0.0]),
                    np.array([0.0, 0.0, length]),
                    tex(seed + 3), 'wall_left'))
    scene.add(Plane(np.array([half, -height / 2, 0.0]),
                    np.array([0.0, height, 0.0]),
                    np.array([0.0, 0.0, length]),
                    tex(seed + 4), 'wall_right'))

    # End wall closes the corridor so there is always structure ahead.
    scene.add(Plane(np.array([-half, -height / 2, length]),
                    np.array([width, 0.0, 0.0]),
                    np.array([0.0, height, 0.0]),
                    tex(seed + 5, c=1.2), 'wall_end'))

    return scene


def add_obstacles(scene, count=8, length=30.0, width=6.0, height=3.0,
                  seed=100, texture_size=256):
    """
    Free-floating panels at varied depths.

    These break the corridor's smooth depth structure, creating occlusion
    boundaries and near-field parallax -- the conditions under which depth
    estimation is actually tested.
    """
    rng = np.random.default_rng(seed)
    for i in range(count):
        z = rng.uniform(3.0, length - 3.0)
        x = rng.uniform(-width / 2 + 0.5, width / 2 - 1.5)
        y = rng.uniform(-height / 2 + 0.3, height / 2 - 1.0)
        w = rng.uniform(0.4, 1.2)
        h = rng.uniform(0.4, 1.2)
        yaw = rng.uniform(-0.5, 0.5)
        u = np.array([w * np.cos(yaw), 0.0, w * np.sin(yaw)])
        v = np.array([0.0, h, 0.0])
        scene.add(Plane(np.array([x, y, z]), u, v,
                        natural_texture(texture_size, alpha=1.1,
                                        seed=seed + i, contrast=1.3),
                        f'obstacle_{i}'))
    return scene
