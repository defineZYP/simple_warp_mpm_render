import os
import sys

import numpy as np
import warp as wp

import torch

from .render import render_frame, vec3_mul, vec3_div
from .camera import Camera
from .scene import init_hdr_image, hdr_process

ACESInputMat = wp.mat33(
    0.59719, 0.35458, 0.04823,
    0.07600, 0.90834, 0.01566,
    0.02840, 0.13383, 0.83777
)

ACESOutputMat = wp.mat33(
    +1.60475, -0.53108, -0.07367,
    -0.10208, +1.10813, -0.00605,
    -0.00327, -0.07276, +1.07602
)

RRT_VEC1 = wp.vec3(0.000090537, 0.000090537, 0.000090537)
RRT_VEC2 = wp.vec3(0.238081, 0.238081, 0.238081)
RRT_VEC3 = wp.vec3(0.4329510, 0.4329510, 0.4329510)
RRT_VEC4 = wp.vec3(0.0245786, 0.0245786, 0.0245786)

@wp.func
def RRTAndODTFit(v: wp.vec3):
    # a = v * (v + RRT_VEC4) - RRT_VEC1
    a = vec3_mul(v, v + RRT_VEC4) - RRT_VEC1
    # b = v * (0.983729 * v + RRT_VEC3) + RRT_VEC2
    b = vec3_mul(v, (0.983729 * v + RRT_VEC3)) + RRT_VEC2
    # return a / b
    return vec3_div(a, b)

@wp.func
def ACESFitted(color: wp.vec3) -> wp.vec3:
    color = ACESInputMat  @ color
    color = RRTAndODTFit(color)
    color = ACESOutputMat @ color
    # return wp.clamp(color, wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 1.0, 1.0))
    return wp.vec3(
        wp.clamp(color.x, 0.0, 1.0),
        wp.clamp(color.y, 0.0, 1.0),
        wp.clamp(color.z, 0.0, 1.0),
    )

@wp.kernel
def render_pixel(
    camera: Camera,
    frame_buffer: wp.array(dtype=wp.vec4),
    frame_pixels: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    buffer = frame_buffer[tid]
    color = wp.vec3(buffer[0] / buffer[3], buffer[1] / buffer[3], buffer[2] / buffer[3])
    color *= camera.exposure
    color = ACESFitted(color)
    # color = wp.pow(color, 1.0 / camera.gamma)
    color = wp.vec3(
        wp.pow(color[0], 1.0 / camera.gamma),
        wp.pow(color[1], 1.0 / camera.gamma),
        wp.pow(color[2], 1.0 / camera.gamma)
    )
    frame_pixels[tid] = color

# 蒙特卡洛采样路径追踪
class PathTracingRender:
    def __init__(
        self,
        cameras=None,
        hdr_path=None,
        sample_per_pixel=256,
        device='cuda:0'
    ):
        self.cameras = cameras
        self.sample_per_pixel = sample_per_pixel
        self.hdr_images = []
        for camera in cameras:
            hdr = init_hdr_image(hdr_path, device=device)
            wp.launch(
                kernel=hdr_process,
                dim=hdr.width * hdr.height,
                inputs=[hdr, camera.exposure, camera.gamma],
                device='cuda:0'
            )
            self.hdr_images.append(hdr)

    def render(
        self,
        camera_id,
        scene,
        scene_materials,
        device='cuda:0'
    ):
        camera = self.cameras[camera_id]
        hdr_image = self.hdr_images[camera_id]
        frame_buffer = wp.zeros(
            shape=camera.num_pixels,
            dtype=wp.vec4,
            device=device
        )
        pixel_buffer = wp.zeros(
            shape=camera.num_pixels,
            dtype=wp.vec3,
            device=device
        )
        wp.launch(
            kernel=render_frame,
            dim=self.sample_per_pixel * camera.num_pixels,
            inputs=[camera, hdr_image, scene, scene_materials],
            outputs=[frame_buffer],
            device=device
        )
        wp.launch(
            kernel=render_pixel,
            dim=camera.num_pixels,
            inputs=[camera, frame_buffer],
            outputs=[pixel_buffer],
            device=device
        )
        return pixel_buffer.numpy()
    