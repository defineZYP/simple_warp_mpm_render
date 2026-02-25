import os
import math
import time
import tqdm
import random
import imageio

import warp as wp
import numpy as np
import matplotlib.pyplot as plt

from constraints.add_collider import add_surface_collider, add_bounding_box
from constraints.add_velocity_translation import add_velocity_on_particles, add_rotation_on_particles
from constraints.modify_gravity import init_stable_fluid_gravity

from engine_utils import *
from mpm_solver_warp import MPM_Simulator_WARP

from render.util import random_renderer

from scene_init import *
from scene_init.materials import materials_range, materials_mapping, materials_trans_mapping
from scene_init.scene_render_info import construct_scene_material_from_materials

from run_template import normal_simulation_once

from get_args import parse_args

wp.init()
wp.config.verify_cuda = True
dvc = "cuda:0"

if __name__ == "__main__":
    # energy conservation
    args = parse_args()
    mode = args.mode
    bias = mode * 100 + 2000
    for idx in tqdm.tqdm(range(1), desc='movies', position=0, leave=False):
        # 随机采样1-3个物体，不给予初速度
        
        scene_info = [
            {
                'instance_type': 'nerd_cube',
                'cube_param': [0.84375, (0.28 - 0.07) * 2, 0.84375],
                'material': 'fluid',
                'center': [0.5, 0.28, 0.5],
                'nerd_center': [0.5, 0.15, 0.5],
                'nerd_cube_param': [0.3, 0.1, 0.3]
            },
            {
                'instance_type': 'cube',
                'cube_param': [0.3, 0.1, 0.3],
                'material': 'gold',
                'center': [0.5, 0.15, 0.5],
            }
            # {
            #     'instance_type': 'cube',
            #     'cube_param': [0.84375, 0.3, 0.84375],
            #     'material': 'red_ink',
            #     'center': [0.5, 0.28 + 0.28 - 0.078125 + 0.2, 0.5]
            # }
        ]

        def stable_init_gravity(
            mpm_solver,
            end_time,
            device,
            *args,
            **kwargs
        ):
            init_stable_fluid_gravity(
                mpm_solver,
                [0.0, -4.9, 0.0],
                end_time=end_time,
                device=device
            )
            add_velocity_on_particles(
                mpm_solver,
                [0., 0., 0.],
                start_time=0.0,
                end_time=end_time,
                mix_type=0,
                region_params=[
                    {
                        'type': 'instance',
                        'param': {
                            'target_instance': 2
                        }
                    }
                ],
                device=device
            )
            
        normal_simulation_once(
            scene_info,
            # save_root='/DATA/DATANAS2/zhangyip/sim_results',
            save_root='./sim_results',
            # idx=idx + bias,
            idx=0,
            # save_r=f'/DATA/DATANAS2/zhangyip/sim_results/{idx + bias}',
            # camera_lookat=(),
            camera_height=0.5,
            num_instances=2,
            warm_up_steps=1600 * 72,
            end_time=1600 * 1e-5 * 72,
            device='cuda:0',
            preprocess=stable_init_gravity
        )
