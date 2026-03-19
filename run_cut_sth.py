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
    # bias = mode * 100 + 1500
    bias = mode * 20 + 1900
    for idx in tqdm.tqdm(range(20), desc='movies', position=0, leave=False):
        # 随机采样1-3个物体，不给予初速度

        scene_info = []
        model_type = random.randint(0, 1)
        material_type = random.randint(0, 2)
        if material_type == 0:
            material = 'jelly'
        elif material_type == 1:
            material = 'plasticine'
        else:
            material = 'foam'
        
        if model_type == 1:
            instance_type = 'cube'
            size = [random.uniform(0.3, 0.5), random.uniform(0.3, 0.5), random.uniform(0.3, 0.5)]
            center = [0.5, 0.078125 + size[1] / 2, 0.5]
            knife_center = [0.4, 0.078125 + size[1] + 0.15, 0.25]
            scene_info.append(
                {
                    'instance_type': 'mesh',
                    'path': './assets/objs/knife.obj',
                    'cube_param': [0.7, 0.7, 0.7],
                    'material': 'iron',
                    'center': knife_center
                }
            )
            scene_info.append(
                {
                    'instance_type': 'cube',
                    'cube_param': size,
                    'material': material,
                    'center': center
                }
            )
        else:
            instance_type = 'mesh'
            size = [random.uniform(0.3, 0.5), random.uniform(0.3, 0.5), random.uniform(0.3, 0.5)]
            center = [0.5, 0.078125 + size[1] / 2, 0.5]
            knife_center = [0.4, 0.078125 + size[1] + 0.15, 0.25]
            mesh_type = random.randint(0, 2)
            if mesh_type == 0:
                mesh_path = "./assets/objs/bread.obj"
                mesh_material = 'bread'
            elif mesh_type == 1:
                mesh_path = "./assets/objs/carrot.obj"
                mesh_material = 'carrot'
            else:
                mesh_path = "./assets/objs/apple.obj"
                mesh_material = 'apple'
            scene_info.append(
                {
                    'instance_type': 'mesh',
                    'path': './assets/objs/knife.obj',
                    'cube_param': [0.7, 0.7, 0.7],
                    'material': 'iron',
                    'center': knife_center
                }
            )
            scene_info.append(
                {
                    'instance_type': 'mesh',
                    'path': mesh_path,
                    'cube_param': size,
                    'material': mesh_material,
                    'center': center,
                    'put_on_bottom': True
                }
            )

        def cut_object(
            mpm_solver,
            bound,
            v_horizon,
            v_vertical,
            delta_t,
            up_bound,
            *args,
            **kwargs
        ):
            start_time = 0.
            position = 0.275
            while start_time <= 1e-5 * 1600 * 120 and position < bound:
                # 刀子向下走 delta_t时间
                add_velocity_on_particles(
                    mpm_solver,
                    [0., -v_horizon, 0.],
                    start_time,
                    start_time + delta_t,
                    mix_type=0,
                    region_params=[
                        {
                            'type': 'instance',
                            'param': {
                                'target_instance': 0
                            }
                        }
                    ],
                    device='cuda:0',
                    after=True
                )

                center_pos = (position + bound) / 2
                seed = random.uniform(0, 1)
                if seed < 0.5:
                    # 刀子把东西向外推delta_t / 5的时间
                    add_velocity_on_particles(
                        mpm_solver,
                        [0, 0, -v_vertical],
                        start_time + delta_t ,
                        start_time + delta_t + delta_t / 3,
                        mix_type=0,
                        region_params=[
                            {
                                'type': 'instance',
                                'param': {
                                    'target_instance': 0
                                }
                            }
                        ],
                        device='cuda:0',
                        after=True
                    )
                    # 刀子向上走delta_t的时间
                    add_velocity_on_particles(
                        mpm_solver,
                        [0, v_horizon, 0],
                        start_time + delta_t + delta_t / 3,
                        start_time + delta_t * 2 + delta_t / 3,
                        mix_type=0,
                        region_params=[
                            {
                                'type': 'instance',
                                'param': {
                                    'target_instance': 0
                                }
                            }
                        ],
                        device='cuda:0',
                        after=True
                    )
                    # 刀子向内回去 delta_t / 3的时间
                    add_velocity_on_particles(
                        mpm_solver,
                        [0, 0, v_vertical],
                        start_time + delta_t * 2 + delta_t / 3,
                        start_time + delta_t * 2 + delta_t / 3 + delta_t * 5 / 6,
                        mix_type=0,
                        region_params=[
                            {
                                'type': 'instance',
                                'param': {
                                    'target_instance': 0
                                }
                            }
                        ],
                        device='cuda:0',
                        after=True
                    )
                    # 整个时间内，不在刀口的物体保持静止，模拟有人在按住这个刀子
                    add_velocity_on_particles(
                        mpm_solver,
                        [0, 0, 0],
                        start_time,
                        start_time + delta_t * 2 + delta_t / 3 + delta_t * 5 / 6,
                        mix_type=1,
                        region_params=[
                            {
                                'type': 'instance',
                                'param': {
                                    'target_instance': 1
                                }
                            },
                            {
                                'type': 'cube',
                                'param': {
                                    'point': [0.5, 0.5, center_pos],
                                    'size': [0.8, 0.8, bound - position]
                                }
                            }
                        ]
                    )
                    start_time += delta_t * 2 + delta_t / 3 + delta_t * 5 / 6
                    position += v_vertical * delta_t / 2
                else:
                    # 刀子直接向上 delta_t
                    add_velocity_on_particles(
                        mpm_solver,
                        [0, v_horizon, 0],
                        start_time + delta_t,
                        start_time + delta_t * 2,
                        mix_type=0,
                        region_params=[
                            {
                                'type': 'instance',
                                'param': {
                                    'target_instance': 0
                                }
                            }
                        ],
                        device='cuda:0',
                        after=True
                    )
                    add_velocity_on_particles(
                        mpm_solver,
                        [0, 0, v_vertical],
                        start_time + delta_t * 2,
                        start_time + delta_t * 2 + delta_t / 2,
                        mix_type=0,
                        region_params=[
                            {
                                'type': 'instance',
                                'param': {
                                    'target_instance': 0
                                }
                            }
                        ],
                        device='cuda:0',
                        after=True
                    )
                    add_velocity_on_particles(
                        mpm_solver,
                        [0, 0, 0],
                        start_time,
                        start_time + delta_t * 2 + delta_t / 2,
                        mix_type=1,
                        region_params=[
                            {
                                'type': 'instance',
                                'param': {
                                    'target_instance': 1
                                }
                            },
                            {
                                'type': 'cube',
                                'param': {
                                    'point': [0.5, 0.5, center_pos],
                                    'size': [0.8, 0.8, bound - position]
                                }
                            }
                        ]
                    )
                    start_time += delta_t * 2 + delta_t / 2
                    position += v_vertical * delta_t / 2
            add_surface_collider(mpm_solver, (0.0, up_bound, 0.0), (0.0, -1.0, 0.0), 'slip', 1.0)
            add_velocity_on_particles(
                mpm_solver,
                [0., 0., 0.],
                start_time,
                999.,
                mix_type=0,
                region_params=[
                    {
                        'type': 'instance',
                        'param': {
                            'target_instance': 0
                        }
                    }
                ],
                device='cuda:0',
                after=True
            )
        
        s = random.randint(0, 3)
        if s == 0:
            delta_t = 1e-5 * 1600 * 12
        elif s == 1:
            delta_t = 1e-5 * 1600 * 18
        elif s == 2:
            delta_t = 1e-5 * 1600 * 24
        else:
            delta_t = 1e-5 * 1600 * 30
        knife_height = knife_center[1] - 0.042 - 0.078125
        v_horizon = knife_height / delta_t
        v_vertical = (center[2] + size[2] / 2 - 0.25) / (2 * delta_t)
        up_bound = min(center[1] + size[1] / 2 + 0.2, 0.921875)
        print(scene_info)
        normal_simulation_once(
            scene_info,
            save_root='/DATA/DATANAS2/zhangyip/sim_results',
            # save_root='./sim_results',
            idx=idx + bias,
            # idx=0,
            # save_r=f'/DATA/DATANAS2/zhangyip/sim_results/{idx + bias}',
            num_instances=2,
            preprocess=cut_object,
            bound=center[2] + size[2] / 2,
            v_horizon=v_horizon,
            v_vertical=v_vertical,
            up_bound=up_bound,
            delta_t=delta_t,
        )