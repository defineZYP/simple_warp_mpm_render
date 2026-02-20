import os
import sys
import torch

import numpy as np
import warp as wp

from mpm_solver_warp import MPM_Simulator_WARP
from mpm_utils import MPMModelStruct, MPMStateStruct

from .selection import *
from .modifier import init_modifier, BaseModifier

# particle_v += force/particle_mass * dt
# this is applied from start_dt, ends after num_dt p2g2p's
# particle velocity is changed before p2g at each timestep
def add_impulse_on_particles(
    solver: MPM_Simulator_WARP,
    force: list,
    dt: float,
    num_dt: int = 1,
    start_time: float = 0.0,
    mix_type: int = False,
    region_params: list = None,
    device: str = "cuda:0"
):
    """
    增加一个脉冲的区域
    """
    if region_params is None:
        region_params = []
    modifier = init_modifier(
        mix_type,
        solver.n_particles,
        device
    )
    modifier.start_time = start_time
    modifier.end_time = start_time + dt * num_dt

    modifier.force = wp.vec3(
        force[0],
        force[1],
        force[2]
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

    # mask = modifier.mask.numpy()
    # print(mask.sum())
    
    solver.impulse_params.append(modifier)

    @wp.kernel
    def apply_force(
        time: float, dt: float, state: MPMStateStruct, param: BaseModifier
    ):
        p = wp.tid()
        if time >= param.start_time and time < param.end_time:
            if param.mask[p] == 1:
                impulse = wp.vec3(
                    param.force[0] / state.particle_mass[p],
                    param.force[1] / state.particle_mass[p],
                    param.force[2] / state.particle_mass[p],
                )
                state.particle_v[p] = state.particle_v[p] + impulse * dt

    solver.pre_p2g_operations.append(apply_force)
