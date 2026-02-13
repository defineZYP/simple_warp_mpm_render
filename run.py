import warp as wp
from mpm_solver_warp import MPM_Simulator_WARP
from engine_utils import *
# from engine_utils_former import *
import torch
wp.init()
wp.config.verify_cuda = True

import tqdm

import imageio

from scene_init import *
from scene_init.materials import materials_range, materials_mapping
from scene_init.scene_render_info import construct_scene_material_from_materials

from constraints.add_collider import add_surface_collider, add_bounding_box
from constraints.add_velocity_translation import add_velocity_on_particles, add_rotation_on_particles

from render.util import random_renderer

import math
import time

import matplotlib.pyplot as plt

dvc = "cuda:0"

if __name__ == "__main__":
    start = time.time()
    scene_name = "ball"
    dt = 1e-5
    # 整个模拟被限制在了(0.1, 0.9)的范围内，防止超限
    # position_vec, velocity_vec, volume_tensor, instances, particles = init_scene(
    #     [
    #         {
    #             'instance_type': 'cube',
    #             'cube_param': [0.8, 0.3, 0.8], 
    #             'material': 'fluid',
    #             'center': [0.5, 0.25, 0.5]
    #         }, 
    #         {
    #             'instance_type': 'ball',
    #             'radius': 0.1, 
    #             'material': 'jelly',
    #             'center': [0.5, 0.8, 0.5]
    #         }
    #     ],
    #     delta_noise_velocity=True,
    # )
    position_vec, velocity_vec, volume_tensor, instances, particles = init_scene(
        [
            {
                'instance_type': 'mesh',
                'path': '/DATA/DATANAS1/zhangyip/phy/warp-mpm/assets/objs/dragon.obj',
                'cube_param': [0.7, 0.7, 0.7],
                'material': 'iron',
                'center': [0.5, 0.5, 0.5]
            },
            # {
            #     'instance_type': 'cube',
            #     'cube_param': [0.5, 0.3, 0.5],
            #     'material': 'fluid',
            #     'center': [0.5, 0.5, 0.5]
            # }
        ]
    )
    # V = 4/3 * math.pi * r ** 3 / particles
    rgb_renderer, optical_renderer = random_renderer()
    mpm_solver = MPM_Simulator_WARP(
        particles, 
        n_grid=256, 
        num_instances=len(instances),
        rgb_renderer=rgb_renderer,
        optical_renderer=optical_renderer
    )
    # volume_tensor = torch.ones(mpm_solver.n_particles) * V
    mpm_solver.load_initial_data_from_torch(
        position_vec.to(dvc),
        velocity_vec.to(dvc),
        volume_tensor.to(dvc),
        num_instances=len(instances)
    )
    material_param = instances[0]['material']
    # mpm_solver.set_parameters_dict(material_param)
    mpm_solver.set_parameters_instances(
        global_kwargs=material_param,
        instances=instances,
    )

    mpm_solver.finalize_mu_lam_bulk()

    # 可弹边界
    add_surface_collider(mpm_solver, (0.0, 0.1, 0.0), (0.0, 1.0, 0.0), 'slip', 1.0)
    add_surface_collider(mpm_solver, (0.0, 0.9, 0.0), (0.0, -1.0, 0.0), 'slip', 1.0)
    add_surface_collider(mpm_solver, (0.1, 0.0, 0.0), (1.0, 0.0, 0.0), 'slip', 1.0)
    add_surface_collider(mpm_solver, (0.9, 0.0, 0.0), (-1.0, 0.0, 0.0), 'slip', 1.0)
    add_surface_collider(mpm_solver, (0.0, 0.0, 0.9), (0.0, 0.0, -1.0), 'slip', 1.0)
    add_surface_collider(mpm_solver, (0.0, 0.0, 0.1), (0.0, 0.0, 1.0), 'slip', 1.0)

    # 防超边界
    add_bounding_box(mpm_solver)

    directory_to_save = f'./sim_results/{scene_name}'

    num_frames = 24 * 5

    head_writer = imageio.get_writer(f'./sim_results/{scene_name}.mp4', fps=24)
    left_writer = imageio.get_writer(f'./sim_results/{scene_name}_left.mp4', fps=24)
    right_writer = imageio.get_writer(f'./sim_results/{scene_name}_right.mp4', fps=24)

    scene_materials = construct_scene_material_from_materials(
        materials_range,
        materials_mapping,
        instances,
        device=dvc
    )

    add_velocity_on_particles(
        mpm_solver,
        [0, 0, 0],
        0.0,
        999.0,
        mix_type=0,
        region_params=[
            {
                'type': 'instance',
                'param': {
                    'target_instance': 0
                }
            }
        ],
        device='cuda:0'
    )

    # add_rotation_on_particles(
    #     mpm_solver,
    #     [0.5, 0.25, 0.5],
    #     [0.0, 1.0, 0.0],
    #     [0.0, 0.0, 0.0],
    #     [1.0, 1.0],
    #     np.pi,
    #     0.0,
    #     0.0,
    #     999.0,
    #     mix_type=0,
    #     region_params=[
    #         {
    #             'type': 'instance',
    #             'param': {
    #                 'target_instance': 0
    #             }
    #         }
    #     ],
    #     device='cuda:0'
    # )

    save_data_at_frame(
        mpm_solver, 
        directory_to_save, 
        0, 
        save_to_ply=False, 
        save_to_h5=False, 
        save_to_video=False, 
        save_to_flow=True,
        dt=dt,
        video_writer=[head_writer, left_writer, right_writer],
        instances=instances,
        scene_materials=scene_materials,
        device=dvc
    )

    J_lis = []
    for i in range(num_frames):
        for k in range(1, 1600):
            # print(mpm_solver.mpm_state.particle_F_trial)
            J = mpm_solver.p2g2p(i, dt, device=dvc)
            J = J.numpy()
            J_lis.append(J[0])
        print(np.any(np.isnan(mpm_solver.mpm_state.particle_x.numpy())))
        save_data_at_frame(
            mpm_solver, 
            directory_to_save, 
            i, 
            save_to_ply=False, 
            save_to_h5=False, 
            save_to_video=True, 
            save_to_flow=True,
            dt=dt,
            video_writer=[head_writer, left_writer, right_writer], 
            instances=instances, 
            scene_materials=scene_materials, 
            device=dvc 
        )

    frames = np.arange(len(J_lis))
    values = np.array(J_lis)

    save_flow_to_file(mpm_solver, directory_to_save)
    # plt.figure(figsize=(10, 6))
    # plt.plot(frames, values, linewidth=2)
    # plt.savefig('./tmp/J_curve.png')

    # mpm_solver.renderer.clear()
    del mpm_solver
    end = time.time()
    print(f"use time {end - start}s")
