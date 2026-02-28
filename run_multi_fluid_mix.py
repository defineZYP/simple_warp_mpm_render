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
from scene_init.materials import materials_range, materials_mapping, materials_trans_mapping, fluid_material_trans_mapping, solid_material_trans_mapping
from scene_init.scene_render_info import construct_scene_material_from_materials

from run_template import normal_simulation_once

from get_args import parse_args

wp.init()
wp.config.verify_cuda = True
dvc = "cuda:0"

def generate_boxes():
    bbox_min = np.array([0.078125, 0.078125, 0.078125])
    bbox_max = np.array([0.921875, 0.625, 0.921875])
    bbox_size = bbox_max - bbox_min
    # axis = np.random.randint(0, 3)
    axis = 1
    split_ratio = np.random.uniform(0.3, 0.45)
    split_point = bbox_min[axis] + split_ratio * bbox_size[axis]

    box1_min = bbox_min.copy()
    box1_max = bbox_max.copy()
    box1_max[axis] = split_point

    box2_min = bbox_min.copy()
    box2_min[axis] = split_point
    box2_max = bbox_max.copy()

    def to_center_size(bmin, bmax):
        center = (bmin + bmax) / 2.0
        size = bmax - bmin
        return center, size
    
    big1_center, big1_size = to_center_size(box1_min, box1_max)
    big2_center, big2_size = to_center_size(box2_min, box2_max)

    if np.random.rand() < 0.5:
        host_min, host_max = box1_min, box1_max
    else:
        host_min, host_max = box2_min, box2_max

    host_size = host_max - host_min

    small_scale = np.random.uniform(0.4, 0.5)
    small_size = host_size * small_scale

    margin = small_size / 2.0
    small_center = np.random.uniform(
        host_min + margin,
        host_max - margin
    )

    return list(big1_center), list(big1_size), list(big2_center), list(big2_size), list(small_center), list(small_size)


if __name__ == "__main__":
    # energy conservation
    args = parse_args()
    error = 61
    mode = args.mode
    bias = mode * 100 + 2000 + (error - 1)
    for idx in tqdm.tqdm(range(100 - (error - 1)), desc='movies', position=0, leave=False):
        center1, size1, center2, size2, center3, size3 = generate_boxes()
        material1 = fluid_material_trans_mapping[random.randint(0, 7)]
        material2 = fluid_material_trans_mapping[random.randint(0, 7)]
        while material1 == material2:
            material2 = fluid_material_trans_mapping[random.randint(0, 7)]
        material3 = solid_material_trans_mapping[random.randint(0, 5)]

        scene_info = [
            {
                'instance_type': 'nerd_cube',
                'cube_param': size1,
                'material': material1,
                'center': center1,
                'nerd_center': center3,
                'nerd_cube_param': size3,
                'velocity': [0., 0., 0.]
            },
            {
                'instance_type': 'nerd_cube',
                'cube_param': size2,
                'material': material2,
                'center': center2,
                'nerd_center': center3,
                'nerd_cube_param': size3,
                'velocity': [0., 0., 0.]
            },
            {
                'instance_type': 'cube',
                'cube_param': size3,
                'material': material3,
                'center': center3,
                'velocity': [0., 0., 0.]
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
            save_root='/DATA/DATANAS2/zhangyip/sim_results',
            # save_root='./sim_results',
            idx=idx + bias,
            # idx=0,
            # save_r=f'/DATA/DATANAS2/zhangyip/sim_results/{idx + bias}',
            # camera_lookat=(),
            camera_height=0.5,
            num_instances=2,
            warm_up_steps=1600 * 24,
            end_time=1600 * 1e-5 * 24,
            device='cuda:0',
            preprocess=stable_init_gravity
        )
