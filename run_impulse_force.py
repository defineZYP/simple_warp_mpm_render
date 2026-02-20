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
from constraints.add_force_translation import add_impulse_on_particles

from engine_utils import *
from mpm_solver_warp import MPM_Simulator_WARP

from render.util import random_renderer

from scene_init import *
from scene_init.materials import materials_range, materials_mapping, materials_trans_mapping
from scene_init.scene_render_info import construct_scene_material_from_materials
from scene_init.util import sample_collide_boxes_3d, sample_boxes_3d

from run_template import normal_simulation_once

from get_args import parse_args

wp.init()
wp.config.verify_cuda = True
dvc = "cuda:0"

solid_material_mapping = {
    0: 'iron',
    1: 'copper',
    2: 'metal',
    3: 'gold',
    4: 'silver',
    5: 'plasticine',
    # 6: 'jelly',
    # 7: 'slime',
    # 8: 'sand',
    # 9: 'foam',
    # 4: 'sand',
}

if __name__ == "__main__":
    # energy conservation
    args = parse_args()
    mode = args.mode
    bias = mode * 100 + 500
    for idx in tqdm.tqdm(range(100), desc='movies', position=0, leave=False):
        # 随机采样1-3个物体，不给予初速度
        source_results, target_results = sample_collide_boxes_3d(
            bounding_box=(0.078125, 0.921875)
        )
        scene_info = []
        source_results_length = len(source_results)
        target_results_length = len(target_results)
        last_obj = 0
        
        for i in range(len(source_results)):
            result = source_results[i]
            center = [result[0], result[1], result[2]]
            size = [result[3], result[4], result[5]]

            material = random.randint(last_obj, len(solid_material_mapping) - source_results_length + i)
            last_obj = material
            material = solid_material_mapping[material]

            _info = {
                'instance_type': 'cube',
                'cube_param': size,
                'material': material,
                'center': center
            }
            scene_info.append(_info)

        last_obj = 0
        
        for i in range(len(target_results)):
            result = target_results[i]
            center = [result[0], result[1], result[2]]
            size = [result[3], result[4], result[5]]

            material = random.randint(last_obj, len(solid_material_mapping) - target_results_length + i)
            last_obj = material
            material = solid_material_mapping[material]

            _info = {
                'instance_type': 'cube',
                'cube_param': size,
                'material': material,
                'center': center
            }
            scene_info.append(_info)

        # 随机选择一个触发方块
        start_idx = random.randint(0, source_results_length - 1)
        material = scene_info[start_idx]['material']
        # print(material)
        material_range = materials_range[materials_mapping[material]]
        density = material_range['density']
        if isinstance(density, tuple):
            density = (density[0] + density[1]) / 2
        cube_param = scene_info[start_idx]['cube_param']
        center = scene_info[start_idx]['center']
        
        def add_impulse_on_instance_obj(
            mpm_solver,
            start_idx,
            density,
            center,
            cube_param,
            instance_device,
            save_path_total,
            total_idx,
            *args,
            **kwargs
        ):
            num_dt = random.randint(500, 2000)
            # print(cube_param)
            mass = density * cube_param[0] * cube_param[1] * cube_param[2]
            # F = mass / (num_dt * 1e-3) 
            F = random.uniform(10, 100)

            # print(F)
            # print(start_idx)
            start_time = random.uniform(0.5 * 0.16, 2.5 * 0.16)
            # start_time = 0.
            save_r = os.path.join(save_path_total, str(total_idx))
            os.makedirs(save_r, exist_ok=True)
            with open(os.path.join(save_r, 'force.txt'), 'w') as f:
                f.write(f"{F} {start_idx} {start_time}")
            cx, cy, cz = center
            x, y, z = cube_param
            x_min = cx - x / 2
            x_max = x_min + x / 4
            cx = (x_min + x_max) / 2

            v = random.uniform(1., 5.)

            # print('inside')

            # add_velocity_on_particles(
            #     mpm_solver,
            #     velocity=[5., 0., 0.],
            #     start_time=start_time,
            #     end_time=start_time + 0.1,
            #     mix_type=True,
            #     region_params=[
            #         {
            #             'type': 'instance',
            #             'param': {
            #                 'target_instance': start_idx
            #             }
            #         },
            #         # {
            #         #     'type': 'cube',
            #         #     'param': {
            #         #         'point': [cx, cy, cz],
            #         #         'size': [x / 4, y, z]
            #         #     }
            #         # }
            #     ],
            #     device=instance_device
            # )

            add_impulse_on_particles(
                mpm_solver,
                force=[F, 0., 0.],
                dt=1e-5,
                num_dt=num_dt,
                start_time=start_time,
                mix_type=True,
                region_params=[
                    {
                        'type': 'instance',
                        'param': {
                            'target_instance': start_idx
                        }
                    },
                    {
                        'type': 'cube',
                        'param': {
                            'point': [cx, cy, cz],
                            'size': [x / 4, y, z]
                        }
                    }
                ],
                device=instance_device
            )
        
        # print(scene_info)
        normal_simulation_once(
            scene_info,
            save_root='/DATA/DATANAS2/zhangyip/sim_results',
            # save_root='./sim_results',
            idx=idx + bias,
            # idx=0,
            preprocess=add_impulse_on_instance_obj,
            dvc="cuda:0",
            start_idx=start_idx,
            density=density,
            center=center,
            cube_param=cube_param,
            instance_device='cuda:0',
            # save_path_total='./sim_results',
            save_path_total='/DATA/DATANAS2/zhangyip/sim_results',
            # total_idx=0
            total_idx=idx+bias
        )