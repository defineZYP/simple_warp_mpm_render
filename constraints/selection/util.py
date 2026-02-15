import os
import sys
import torch

import numpy as np
import warp as wp

# from mpm_solver_warp import MPM_Simulator_WARP
from mpm_utils import MPMModelStruct, MPMStateStruct

from .cube_selection import selection_particle_cube_region
from .cylinder_selection import selection_particle_cylinder_region
from .instance_selection import selection_particle_instance_region

from ..modifier import BaseModifier

"""
For cube_selection:
```
params = {
    "point": [1., 1., 1.],
    "size": [1., 1., 1.],
}
```

For cylinder_selection:
```
params = {
    "point": [1., 1., 1.],
    "normal": [1., 1., 1.],
    "half_height_and_radius": 1.
}
```

For instance_selection:
```
params = {
    "target_instance": 1
}
```
"""

def standard_select_region(
    region_type: str,
    n_particles: int,
    state: MPMStateStruct,
    modifier: BaseModifier,
    mix_type: int,
    device: str,
    params: dict,
):
    if params is None:
        params = {}
    if region_type == "cube":
        point = wp.vec3(
            params['point'][0],
            params['point'][1],
            params['point'][2],
        )
        size = wp.vec3(
            params['size'][0],
            params['size'][1],
            params['size'][2],
        )
        wp.launch(
            kernel=selection_particle_cube_region,
            dim=n_particles,
            inputs=[
                state,
                modifier,
                point,
                size,
                mix_type
            ],
            device=device
        )
    elif region_type == "cylinder":
        point = wp.vec3(
            params['point'][0],
            params['point'][1],
            params['point'][2],
        )
        normal = wp.vec3(
            params['normal'][0],
            params['normal'][1],
            params['normal'][2],
        )
        half_height_and_radius = wp.vec2(
            params['half_height_and_radius'][0],
            params['half_height_and_radius'][1]
        )
        wp.launch(
            kernel=selection_particle_cylinder_region,
            dim=n_particles,
            inputs=[
                state,
                modifier,
                point,
                normal,
                half_height_and_radius,
                mix_type
            ],
            device=device
        )
    elif region_type == "instance":
        wp.launch(
            kernel=selection_particle_instance_region,
            dim=n_particles,
            inputs=[
                state,
                modifier,
                params['target_instance'],
                mix_type
            ],
            device=device
        )
    else:
        raise NotImplementedError(f"Type {region_type} is not supported yet.")
    return params