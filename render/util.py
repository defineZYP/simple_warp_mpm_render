import os
import math
import random
import numpy as np

from .camera import init_camera, init_head_camera
from .renderer import PathTracingRender
from .optical_flow_renderer import OpticalFlowRenderer

def init_renderer():
    pass

def random_renderer(
    fov=40,
    width=1280,
    height=720,
    exposure=1.0,
    aperture=0.0,
    focus_distance=0.1,
    near=0.1,
    gamma=2.2,
    device='cuda:0'
):
    center = (0.5, 0.3875, 0.5)

    # R = 1.875
    R = 1.325

    theta = random.uniform(0, 2) * math.pi

    camera_x = 0.5 + R * math.cos(theta)
    camera_z = 0.5 + R * math.sin(theta)
    camera_pos = (camera_x, 1.25, camera_z)
    camera_front = [center[i] - camera_pos[i] for i in range(3)]
    camera_front /= np.linalg.norm(camera_front)
    camera_front = tuple(camera_front)

    cameras = init_head_camera(
        position=camera_pos,
        lookat=center,
        up=(0.0, 1.0, 0.0),
        fov=fov,
        width=width,
        height=height,
        exposure=exposure,
        aperture=aperture,
        focus_distance=focus_distance,
        near=near,
        gamma=gamma
    )

    hdr_id = random.randint(0, 31)
    renderer = PathTracingRender(
        cameras=cameras,
        hdr_path=f"./assets/HDRi/{hdr_id}.hdr",
        sample_per_pixel=8,
        device=device
    )

    optical_renderer = OpticalFlowRenderer(
        cameras=[
            init_camera(
                position=camera_pos,
                lookat=center,
                up=(0.0, 1.0, 0.0),
                fov=fov,
                width=width,
                height=height,
                exposure=exposure,
                aperture=aperture,
                focus_distance=focus_distance,
                near=near,
                gamma=gamma
            )
        ],
        device=device
    )

    return renderer, optical_renderer
