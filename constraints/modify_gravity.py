import os
import sys
import torch

import numpy as np
import warp as wp

from mpm_solver_warp import MPM_Simulator_WARP
from mpm_utils import MPMModelStruct, MPMStateStruct

from .selection import *
from .modifier import init_modifier, BaseModifier

def init_stable_fluid_gravity(
    solver: MPM_Simulator_WARP,
    gravity: list,
    end_time: float = 999.0,
    device: str = "cuda:0"
):
    modifier = init_modifier(
        0,
        1,
        device
    )
    modifier.start_time = 0.
    modifier.end_time = end_time
    # change_times = end_time / dt
    modifier.velocity = wp.vec3(
        gravity[0] / end_time,
        gravity[1] / end_time,
        gravity[2] / end_time
    )

    solver.model_param_modifier_params.append(modifier)

    @wp.kernel
    def modify_model_param(
        time: float,
        model: MPMModelStruct,
        modifier_params: BaseModifier
    ):
        p = wp.tid()
        if time < modifier_params.end_time:
            model.gravitational_accelaration = modifier_params.velocity * time

    solver.model_param_modifiers.append(modify_model_param)