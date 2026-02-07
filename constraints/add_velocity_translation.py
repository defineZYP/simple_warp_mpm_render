import os
import sys
import torch

import numpy as np
import warp as wp

from mpm_solver_warp import MPM_Simulator_WARP
from mpm_utils import MPMModelStruct, MPMStateStruct

from .selection import *
from .modifier import init_modifier, BaseModifier

def add_velocity_on_particles(
    solver: MPM_Simulator_WARP,
    velocity: list,
    start_time: float = 0.0,
    end_time: float = 999.0,
    mix_type: int = False,
    region_params: list = None,
    device: str = "cuda:0"
):
    """
    给点增加一个速度，本身不受MPM里其他点作用，相当于有一个外力出现支持该质点出现这些物理现象
    """
    if region_params is None:
        region_params = []
    
    modifier = init_modifier(
        mix_type,
        solver.n_particles,
        device
    )
    modifier.start_time = start_time
    modifier.end_time = end_time
    modifier.velocity = wp.vec3(
        velocity[0],
        velocity[1],
        velocity[2]
    )

    for region_param in region_params:
        region_type = region_param['type']
        params = region_param['param']
        standard_select_region(
            region_type=region_type,
            n_particles=solver.n_particles,
            state=solver.mpm_state,
            modifier=modifier,
            mix_type=mix_type,
            device=device,
            params=params
        )

    solver.particle_velocity_modifier_params.append(modifier)

    @wp.kernel
    def modify_particle_v_before_p2g(
        time: float,
        state: MPMStateStruct,
        velocity_modifier_params: BaseModifier
    ):
        p = wp.tid()
        if time >= velocity_modifier_params.start_time and time < velocity_modifier_params.end_time:
            if velocity_modifier_params.mask[p] == 1:
                state.particle_v[p] = velocity_modifier_params.velocity
    
    solver.particle_velocity_modifiers.append(modify_particle_v_before_p2g)

def add_rotation_on_particles(
    solver: MPM_Simulator_WARP,
    point: list,
    normal: list,
    direction: list,
    half_height_and_radius: list,
    rotation_scale: float,
    translation_scale: float,
    start_time: float = 0.0,
    end_time: float = 0.0,
    mix_type: int = False,
    region_params: list = None,
    device: str = "cuda:0"
):
    normal_scale = 1.0 / wp.sqrt(float(normal[0]**2 + normal[1]**2 + normal[2]**2))
    normal = list(normal_scale * x for x in normal)
    modifier = init_modifier(
        mix_type,
        solver.n_particles,
        device
    )

    modifier.point = wp.vec3(
        point[0],
        point[1],
        point[2]
    )
    modifier.normal = wp.vec3(
        normal[0],
        normal[1],
        normal[2]
    )
    modifier.direction = wp.vec3(
        direction[0],
        direction[1],
        direction[2]
    )
    horizontal_1 = wp.vec3(1.0,1.0,1.0)
    if wp.abs(wp.dot(modifier.normal, horizontal_1)) < 0.01:
        horizontal_1 = wp.vec3(0.72, 0.37, -0.67)
    horizontal_1 = horizontal_1 - wp.dot(horizontal_1, modifier.normal) * modifier.normal
    horizontal_1 = horizontal_1 * (1.0 / wp.length(horizontal_1))
    horizontal_2 = wp.cross(horizontal_1, modifier.normal)

    modifier.horizontal_axis_1 = horizontal_1
    modifier.horizontal_axis_2 = horizontal_2

    modifier.rotation_scale = rotation_scale
    modifier.translation_scale = translation_scale

    modifier.start_time = start_time
    modifier.end_time = end_time

    for region_param in region_params:
        region_type = region_param['type']
        params = region_param['param']
        standard_select_region(
            region_type=region_type,
            n_particles=solver.n_particles,
            state=solver.mpm_state,
            modifier=modifier,
            mix_type=mix_type,
            device=device,
            params=params
        )
    
    solver.particle_velocity_modifier_params.append(modifier)

    @wp.kernel
    def modify_particle_v_before_p2g(
        time: float,
        state: MPMStateStruct,
        velocity_modifier_params: BaseModifier
    ):
        p = wp.tid()
        if time >= velocity_modifier_params.start_time and time < velocity_modifier_params.end_time:
            if velocity_modifier_params.mask[p] == 1:
                offset = state.particle_x[p] - velocity_modifier_params.point
                horizontal_distance = wp.length(offset - wp.dot(offset, velocity_modifier_params.normal) * velocity_modifier_params.normal)
                cosine = wp.dot(offset, velocity_modifier_params.horizontal_axis_1) / horizontal_distance
                theta = wp.acos(cosine)
                if wp.dot(offset, velocity_modifier_params.horizontal_axis_2) > 0:
                    theta = theta
                else:
                    theta = -theta
                axis1_scale = - horizontal_distance * wp.sin(theta) * velocity_modifier_params.rotation_scale
                axis2_scale = horizontal_distance * wp.cos(theta) * velocity_modifier_params.rotation_scale
                axis_vertical_scale = translation_scale
                state.particle_v[p] = axis1_scale * velocity_modifier_params.horizontal_axis_1 + axis2_scale * velocity_modifier_params.horizontal_axis_2 + axis_vertical_scale * velocity_modifier_params.direction
    
    solver.particle_velocity_modifiers.append(modify_particle_v_before_p2g)
