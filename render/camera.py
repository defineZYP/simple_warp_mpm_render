import os
import sys
import torch

import numpy as np
import warp as wp

from .ray import Ray

@wp.struct
class Camera:
    position: wp.vec3
    lookat: wp.vec3
    up: wp.vec3
    forward: wp.vec3
    right: wp.vec3
    fov: float
    width: float
    height: float
    num_pixels: int
    exposure: float
    aperture: float
    lens_radius: float
    focus_distance: float
    near: float
    aspect: float
    plane_height: float
    plane_width: float
    gamma: float

def init_camera(
    position=(0,0,0),
    lookat=(1,0,0),
    up=(0,1,0),
    fov=35,
    width=512,
    height=512,
    exposure=1.0,
    aperture=0.0,
    focus_distance=1.0,
    near=1.0,
    gamma=2.2
):
    position = np.array(position, dtype=np.float32)
    lookat = np.array(lookat, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    forward = lookat - position
    forward /= np.linalg.norm(forward)
    
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    up = np.cross(right, forward)

    aspect = width / height
    fov_rad = np.deg2rad(fov)

    plane_height = 2.0 * np.tan(fov_rad * 0.5) * near
    plane_width = plane_height * aspect

    camera = Camera()
    camera.gamma = gamma
    camera.num_pixels = int(height + 0.5) * int(width + 0.5)
    camera.position = wp.vec3(position[0], position[1], position[2])
    camera.lookat = wp.vec3(lookat[0], lookat[1], lookat[2])
    camera.up = wp.vec3(up[0], up[1], up[2])
    camera.forward = wp.vec3(forward[0], forward[1], forward[2])
    camera.right = wp.vec3(right[0], right[1], right[2])
    camera.aspect = float(aspect)
    camera.plane_height = float(plane_height)
    camera.plane_width = float(plane_width)
    camera.exposure = float(exposure)
    camera.aperture = float(aperture)
    camera.lens_radius = camera.aperture * 0.5
    camera.focus_distance = float(focus_distance)
    camera.near = float(near)
    camera.fov = float(fov)

    camera.width = float(width)
    camera.height = float(height)

    return camera

def init_head_camera(
    position=(0,0,0),
    lookat=(1,0,0),
    up=(0,1,0),
    fov=35,
    width=512,
    height=512,
    exposure=1.0,
    aperture=0.0,
    focus_distance=1.0,
    near=1.0,
    gamma=2.2,
    IPD=0.064
):
    head_cam = init_camera(
        position,
        lookat,
        up,
        fov,
        width,
        height,
        exposure,
        aperture,
        focus_distance,
        near,
        gamma
    )
    # normalize之后的结果
    head_position = np.array(head_cam.position)
    head_lookat = np.array(head_cam.lookat)
    head_up = np.array(head_cam.up)
    head_right = np.array(head_cam.right)

    d = IPD / 2
    left_position = head_position - d * head_right
    left_cam = init_camera(
        left_position,
        head_lookat,
        head_up,
        fov,
        width,
        height,
        exposure,
        aperture,
        focus_distance,
        near,
        gamma
    )

    right_position = head_position + d * head_right
    right_cam = init_camera(
        right_position,
        head_lookat,
        head_up,
        fov,
        width,
        height,
        exposure,
        aperture,
        focus_distance,
        near,
        gamma
    )

    return [head_cam, left_cam, right_cam]

@wp.func
def _sample_disk(tid: int):
    r = wp.sqrt(wp.randf(wp.uint32(tid + 1)))
    theta = 2.0 * wp.pi * wp.randf(wp.uint32(tid + 2))
    return r * wp.cos(theta), r * wp.sin(theta)

@wp.func
def generate_ray(c: Camera, px: float, py: float, tid: int):
    # -------- pinhole ray ---------
    u = (px + 0.5) / c.width - 0.5
    v = (py + 0.5) / c.height - 0.5

    x = u * c.plane_width
    y = -v * c.plane_height

    pinhole_dir = (
        c.forward * c.near + c.right * x + c.up * y
    )

    pinhole_dir /= wp.length(pinhole_dir)
    
    # ------- thin lens -----------
    if c.lens_radius > 0.0:
        t = c.focus_distance / wp.dot(pinhole_dir, c.forward)
        focus_point = c.position + pinhole_dir * t
        dx, dy = _sample_disk(tid)
        dx = dx * c.lens_radius
        dy = dy * c.lens_radius
        lens_offset = c.right * dx + c.up * dy
        origin = c.position + lens_offset
        direction = focus_point - origin
        direction /= wp.length(direction)
    else:
        origin = c.position
        direction = pinhole_dir
    return Ray(origin=origin, direction=direction, color=wp.vec3(1.0, 1.0, 1.0), depth=0)

if __name__ == "__main__":
    camera = init_camera(width=4, height=4)
    print(camera)
    @wp.kernel
    def test_sample_ray(c: Camera):
        tid = wp.tid()
        height = wp.int32(c.height)
        x = float(tid % height)
        y = float(tid // height)
        ray = generate_ray(c, x, y, tid)

    wp.launch(
        kernel=test_sample_ray,
        dim=int(camera.height * camera.width),
        inputs=[camera],
        device='cuda:0'
    )
