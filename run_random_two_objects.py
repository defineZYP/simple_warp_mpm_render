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
    args = parse_args()
    mode = args.mode
    bias = mode * 200
    for idx in tqdm.tqdm(range(200), desc='movies', position=0):
        # 随机采样1-3个物体，不给予初速度
        items = random.randint(1, 3)
        results = sample_boxes_3d(
            items
        )
        scene_info = []
        for i in range(items):
            result = results[i]
            center = [result[0], result[1], result[2]]
            size = [result[3], result[4], result[5]]

            material = random.randint(0, 12)
            material = materials_trans_mapping[material]

            obj_type = random.randint(0, 2)
            if obj_type == 0:
                # sphere
                radius = np.min(size)
                _info = {
                    'instance_type': 'ball',
                    'radius': radius,
                    'material': material,
                    'center': center
                }
                scene_info.append(_info)
            elif obj_type == 1:
                # cube
                _info = {
                    'instance_type': 'cube',
                    'cube_param': size,
                    'material': material,
                    'center': center
                }
                scene_info.append(_info)
            else:
                mesh_name = random.randint(0, 7)
                # mesh_name = 8
                mesh_name = mesh_mapping[mesh_name]
                _info = {
                    'instance_type': 'mesh',
                    'path': f'./assets/objs/{mesh_name}.obj',
                    'cube_param': size,
                    'material': material,
                    'center': center
                }
                scene_info.append(_info)
        
        print(scene_info)
        normal_simulation_once(
            scene_info,
            save_root='/DATA/DATANAS2/zhangyip/sim_results',
            idx=idx + bias,
        )