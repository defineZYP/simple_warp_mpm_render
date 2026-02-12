import os
import sys

import numpy as np
import warp as wp

import torch

from .camera import Camera

from warp_utils import MPMStateStruct
from scipy.ndimage import uniform_filter, convolve

def average_particles(img, mask_value, kernel=(5,5)):
    kernel = np.ones(kernel, dtype=np.float32)

    out = img.copy()

    for ch in range(img.shape[2]):
        I = img[:,:,ch]

        mask = (I != mask_value).astype(np.float32)

        value_sum = convolve(I * mask, kernel, mode='reflect')
        weight_sum = convolve(mask, kernel, mode='reflect')

        avg = np.divide(value_sum, weight_sum,
                        out=np.zeros_like(value_sum),
                        where=weight_sum>0)

        # 只替换原本为 0 的位置
        out[:,:,ch][I == mask_value] = avg[I == mask_value]

    return out

@wp.func
def projection(
    camera: Camera,
    v: wp.vec3
):
    return wp.vec3(
        wp.dot(v, camera.right),
        wp.dot(v, camera.up),
        wp.dot(v, camera.forward),
    )

@wp.kernel
def render_flow(
    camera: Camera,
    state: MPMStateStruct,
    z_buffer: wp.array(dtype=float),
    flow_buffer: wp.array(dtype=wp.vec2),
    f_buffer: wp.array(dtype=wp.vec2),
    dt: float
):
    tid = wp.tid()
    p_x = state.particle_x[tid]
    p_v = state.particle_v[tid]
    p_f = state.particle_tf[tid] * state.particle_mass[tid] / dt

    # projection
    p_x = projection(camera, p_x - camera.position)
    p_v = projection(camera, p_v)    
    p_f = projection(camera, p_f)

    scale = camera.near / p_x[2]

    ix = int((p_x[0] * scale / camera.plane_width + 0.5) * camera.width)
    iy = int((0.5 - p_x[1] * scale / camera.plane_height) * camera.height)  # 图像坐标y向下

    if not (ix < 0 or iy < 0 or ix >= camera.width or iy >= camera.height):
        if p_x.z > 1e-6:
            pixel = iy * int(camera.width) + ix
            f = camera.focus_distance
            inv_z2 = 1.0 / (p_x[2] ** 2.)
            du_v = f * (p_v[0] * p_x[2] - p_x[0] * p_v[2]) * inv_z2
            dv_v = f * (p_v[1] * p_x[2] - p_x[1] * p_v[2]) * inv_z2

            du_f = f * (p_f[0] * p_x[2] - p_x[0] * p_f[2]) * inv_z2
            dv_f = f * (p_f[1] * p_x[2] - p_x[1] * p_f[2]) * inv_z2

            old_depth = wp.atomic_min(z_buffer, pixel, p_x[2])
            if p_x[2] <= old_depth:
                flow_buffer[pixel] = wp.vec2(du_v, -dv_v)
                f_buffer[pixel] = wp.vec2(du_f, -dv_f)

class OpticalFlowRenderer:
    def __init__(
        self,
        cameras=None,
        device='cuda:0'
    ):
        self.cameras = cameras
        self.device = device
        self.flows = []
        self.forces = []
        self.depths = []

    def render(
        self,
        camera_id,
        n_particles,
        mpm_state,
        dt,
        device='cuda:0',
    ):
        camera = self.cameras[camera_id]
        z_buffer = wp.zeros(
            shape=camera.num_pixels,
            dtype=float,
            device=device
        ) + 1e10
        flow_buffer = wp.zeros(
            shape=camera.num_pixels,
            dtype=wp.vec2,
            device=device
        )
        f_buffer = wp.zeros(
            shape=camera.num_pixels,
            dtype=wp.vec2,
            device=device
        )
        # print(n_particles)
        wp.launch(
            kernel=render_flow,
            dim=n_particles,
            inputs=[camera, mpm_state, z_buffer, flow_buffer, f_buffer, dt],
            device=device
        )
        self.flows.append(
            average_particles(
                flow_buffer.numpy().reshape(int(camera.height), int(camera.width), -1),
                mask_value=0
            )
        )
        self.forces.append(
            average_particles(
                f_buffer.numpy().reshape(int(camera.height), int(camera.width), -1),
                mask_value=0
            )
        )
        self.depths.append(
            average_particles(
                z_buffer.numpy().reshape(int(camera.height), int(camera.width), -1),
                mask_value=1e10
            )
        )
        # return z_buffer.numpy(), flow_buffer.numpy(), f_buffer.numpy()
