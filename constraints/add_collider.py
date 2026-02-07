import os
import sys
import torch

import numpy as np
import warp as wp

from .collider import Dirichlet_collider
from mpm_solver_warp import MPM_Simulator_WARP
from mpm_utils import MPMModelStruct, MPMStateStruct

# a surface specified by a point and the normal vector
def add_surface_collider(
    solver: MPM_Simulator_WARP,
    point: list,
    normal: list,
    surface: str = "sticky",
    friction: float = 0.0,
    start_time: float = 0.0,
    end_time: float = 999.0,
):
    point = list(point)
    # Normalize normal
    normal_scale = 1.0 / wp.sqrt(float(sum(x**2 for x in normal)))
    normal = list(normal_scale * x for x in normal)

    collider_param = Dirichlet_collider()
    collider_param.start_time = start_time
    collider_param.end_time = end_time

    collider_param.point = wp.vec3(point[0], point[1], point[2])
    collider_param.normal = wp.vec3(normal[0], normal[1], normal[2])

    if surface == "sticky" and friction != 0:
        raise ValueError("friction must be 0 on sticky surfaces.")
    if surface == "sticky":
        collider_param.surface_type = 0
    elif surface == "slip":
        collider_param.surface_type = 1
    elif surface == "cut":
        collider_param.surface_type = 11
    else:
        collider_param.surface_type = 2
    # frictional
    collider_param.friction = friction

    solver.collider_params.append(collider_param)

    @wp.kernel
    def collide(
        time: float,
        dt: float,
        state: MPMStateStruct,
        model: MPMModelStruct,
        param: Dirichlet_collider,
    ):
        grid_x, grid_y, grid_z = wp.tid()
        if time >= param.start_time and time < param.end_time:
            offset = wp.vec3(
                float(grid_x) * model.dx - param.point[0],
                float(grid_y) * model.dx - param.point[1],
                float(grid_z) * model.dx - param.point[2],
            )
            n = wp.vec3(param.normal[0], param.normal[1], param.normal[2])
            dotproduct = wp.dot(offset, n)

            if dotproduct < 0.0:
                if param.surface_type == 0:
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                        0.0, 0.0, 0.0
                    )
                elif param.surface_type == 11:
                    if (
                        float(grid_z) * model.dx < 0.4
                        or float(grid_z) * model.dx > 0.53
                    ):
                        state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                            0.0, 0.0, 0.0
                        )
                    else:
                        v_in = state.grid_v_out[grid_x, grid_y, grid_z]
                        state.grid_v_out[grid_x, grid_y, grid_z] = (
                            wp.vec3(v_in[0], 0.0, v_in[2]) * 0.3
                        )
                else:
                    v = state.grid_v_out[grid_x, grid_y, grid_z]
                    normal_component = wp.dot(v, n)
                    if param.surface_type == 1:
                        v = (
                            v - normal_component * n
                        )  # Project out all normal component
                    else:
                        v = (
                            v - wp.min(normal_component, 0.0) * n
                        )  # Project out only inward normal component
                    if normal_component < 0.0 and wp.length(v) > 1e-20:
                        v = wp.max(
                            0.0, wp.length(v) + normal_component * param.friction
                        ) * wp.normalize(
                            v
                        )  # apply friction here
                    # state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                    #     0.0, 0.0, 0.0
                    # )
                    state.grid_v_out[grid_x, grid_y, grid_z] = v

    solver.grid_postprocess.append(collide)
    solver.modify_bc.append(None)

# a cubiod is a rectangular cube'
# centered at `point`
# dimension is x: point[0]±size[0]
#              y: point[1]±size[1]
#              z: point[2]±size[2]
# all grid nodes lie within the cubiod will have their speed set to velocity
# the cuboid itself is also moving with const speed = velocity
# set the speed to zero to fix BC
def set_velocity_on_cuboid(
    solver: MPM_Simulator_WARP,
    point: list,
    size: list,
    velocity: float,
    start_time: float = 0.0,
    end_time: float = 999.0,
    reset: int = 0,           # actually bool may be better
):
    point = list(point)

    collider_param = Dirichlet_collider()
    collider_param.start_time = start_time
    collider_param.end_time = end_time
    collider_param.point = wp.vec3(point[0], point[1], point[2])
    collider_param.size = size
    collider_param.velocity = wp.vec3(velocity[0], velocity[1], velocity[2])
    # collider_param.threshold = threshold
    collider_param.reset = reset
    solver.collider_params.append(collider_param)

    @wp.kernel
    def collide(
        time: float,
        dt: float,
        state: MPMStateStruct,
        model: MPMModelStruct,
        param: Dirichlet_collider,
    ):
        grid_x, grid_y, grid_z = wp.tid()
        if time >= param.start_time and time < param.end_time:
            offset = wp.vec3(
                float(grid_x) * model.dx - param.point[0],
                float(grid_y) * model.dx - param.point[1],
                float(grid_z) * model.dx - param.point[2],
            )
            if (
                wp.abs(offset[0]) < param.size[0]
                and wp.abs(offset[1]) < param.size[1]
                and wp.abs(offset[2]) < param.size[2]
            ):
                state.grid_v_out[grid_x, grid_y, grid_z] = param.velocity
        elif param.reset == 1:
            if time < param.end_time + 15.0 * dt:
                state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)

    def modify(time, dt, param: Dirichlet_collider):
        if time >= param.start_time and time < param.end_time:
            param.point = wp.vec3(
                param.point[0] + dt * param.velocity[0],
                param.point[1] + dt * param.velocity[1],
                param.point[2] + dt * param.velocity[2],
            )  # param.point + dt * param.velocity

    solver.grid_postprocess.append(collide)
    solver.modify_bc.append(modify)

def add_bounding_box(
    solver: MPM_Simulator_WARP, 
    start_time: float = 0.0, 
    end_time: float = 999.0
):
    collider_param = Dirichlet_collider()
    collider_param.start_time = start_time
    collider_param.end_time = end_time

    solver.collider_params.append(collider_param)

    @wp.kernel
    def collide(
        time: float,
        dt: float,
        state: MPMStateStruct,
        model: MPMModelStruct,
        param: Dirichlet_collider,
    ):
        grid_x, grid_y, grid_z = wp.tid()
        padding = 3
        if time >= param.start_time and time < param.end_time:
            if grid_x < padding and state.grid_v_out[grid_x, grid_y, grid_z][0] < 0:
                state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                    0.0,
                    state.grid_v_out[grid_x, grid_y, grid_z][1],
                    state.grid_v_out[grid_x, grid_y, grid_z][2],
                )
            if (
                grid_x >= model.grid_dim_x - padding
                and state.grid_v_out[grid_x, grid_y, grid_z][0] > 0
            ):
                state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                    0.0,
                    state.grid_v_out[grid_x, grid_y, grid_z][1],
                    state.grid_v_out[grid_x, grid_y, grid_z][2],
                )

            if grid_y < padding and state.grid_v_out[grid_x, grid_y, grid_z][1] < 0:
                state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                    state.grid_v_out[grid_x, grid_y, grid_z][0],
                    0.0,
                    state.grid_v_out[grid_x, grid_y, grid_z][2],
                )
            if (
                grid_y >= model.grid_dim_y - padding
                and state.grid_v_out[grid_x, grid_y, grid_z][1] > 0
            ):
                state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                    state.grid_v_out[grid_x, grid_y, grid_z][0],
                    0.0,
                    state.grid_v_out[grid_x, grid_y, grid_z][2],
                )

            if grid_z < padding and state.grid_v_out[grid_x, grid_y, grid_z][2] < 0:
                state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                    state.grid_v_out[grid_x, grid_y, grid_z][0],
                    state.grid_v_out[grid_x, grid_y, grid_z][1],
                    0.0,
                )
            if (
                grid_z >= model.grid_dim_z - padding
                and state.grid_v_out[grid_x, grid_y, grid_z][2] > 0
            ):
                state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(
                    state.grid_v_out[grid_x, grid_y, grid_z][0],
                    state.grid_v_out[grid_x, grid_y, grid_z][1],
                    0.0,
                )

    solver.grid_postprocess.append(collide)
    solver.modify_bc.append(None)
