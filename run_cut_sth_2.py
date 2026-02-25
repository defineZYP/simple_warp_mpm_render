import random
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

def up_and_down(mpm_solver, v, position, bound, start_time):
    delta = 24. / v

    add_velocity_on_particles(
        mpm_solver,
        [0, -v, 0],
        start_time,
        start_time + 1e-5 * 1600 * delta,
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
        add_velocity_on_particles(
            mpm_solver,
            [0, 0, -v/5],
            start_time + 1e-5 * 1600 * delta,
            start_time + 1e-5 * 1600 * delta * 1.25,
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
            # after=True
        )

        add_velocity_on_particles(
            mpm_solver,
            [0, 0, v/5],
            start_time + 1e-5 * 1600 * delta * 1.25,
            start_time + 1e-5 * 1600 * delta * 1.5,
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
            # after=True
        )

        add_velocity_on_particles(
            mpm_solver,
            [0, 0, 0],
            start_time,
            start_time + 1e-5 * 1600 * delta * 2.2 + 1e-5 * 1600 * delta * 0.5,
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

        start_time += 1e-5 * 1600 * delta * 0.5
    else:
        add_velocity_on_particles(
            mpm_solver,
            [0, 0, 0],
            start_time,
            start_time + 1e-5 * 1600 * delta * 2.2,
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
        
    add_velocity_on_particles(
        mpm_solver,
        [0, v, 0],
        start_time + 1e-5 * 1600 * delta,
        start_time + 1e-5 * 1600 * delta * 2,
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

    # 稍微往左边带一点
    add_velocity_on_particles(
        mpm_solver,
        [0, 0, v],
        start_time + 1e-5 * 1600 * delta * 2,
        start_time + 1e-5 * 1600 * delta * 2.2,
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

    # print(bound - position)
    print(center_pos, position, bound)
    # add_velocity_on_particles(
    #     mpm_solver,
    #     [0, 0, 0],
    #     start_time,
    #     start_time + 1e-5 * 1600 * delta * 2.2,
    #     mix_type=1,
    #     region_params=[
    #         {
    #             'type': 'instance',
    #             'param': {
    #                 'target_instance': 1
    #             }
    #         },
    #         {
    #             'type': 'cube',
    #             'param': {
    #                 'point': [0.5, 0.5, center_pos],
    #                 'size': [0.8, 0.8, bound - position]
    #             }
    #         }
    #     ]
    # )

    return start_time + 1e-5 * 1600 * delta * 2.2, position + 1e-5 * 1600 * delta * 0.2 * v

def cut_object(
    mpm_solver,
    bound,
    v_horizon,
    v_vertical,
    delta_t,
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

    # position_vec, velocity_vec, volume_tensor, instances, particles = init_scene(
    #     [
    #         {
    #             'instance_type': 'mesh',
    #             'path': './assets/objs/knife.obj',
    #             'cube_param': [0.7, 0.7, 0.7],
    #             'material': 'iron',
    #             'center': [0.5, 0.078125 + 0.24375 + 0.15, 0.2] # [0.5, 0.5, 0.2] [0.14995966 0.45870931 0.18639484] [0.85004034 0.54129069 0.21360516]
    #         },
    #         # {
    #         #     'instance_type': 'mesh',
    #         #     'path': './assets/objs/carrot.obj',
    #         #     'cube_param': [0.5, 0.24375, 0.5],
    #         #     'material': 'carrot',
    #         #     'put_on_bottom': True,
    #         #     # "material": "jelly",
    #         #     'center': [0.5, 0.2, 0.5]
    #         # }
    #         # {
    #         #     'instance_type': 'mesh',
    #         #     'path': './assets/objs/apple.obj',
    #         #     'cube_param': [0.5, 0.34375, 0.5],
    #         #     'material': 'apple',
    #         #     'put_on_bottom': True,
    #         #     # "material": "jelly",
    #         #     'center': [0.5, 0.25, 0.5]
    #         # }
    #         {
    #             'instance_type': 'mesh',
    #             'path': './assets/objs/bread.obj',
    #             'cube_param': [0.5, 0.24375, 0.5],
    #             'material': 'bread',
    #             'put_on_bottom': True,
    #             # "material": "jelly",
    #             'center': [0.5, 0.2, 0.5]
    #         }
    #     ]
    # )

    position_vec, velocity_vec, volume_tensor, instances, particles = init_scene(
        [
            {
                'instance_type': 'mesh', 
                'path': './assets/objs/knife.obj', 
                'cube_param': [0.8, 0.8, 0.8], 
                'material': 'iron', 
                'center': [0.4, 0.5688124985931406, 0.25]
            }, 
            {
                'instance_type': 'cube', 
                'cube_param': [0.38562015555224183, 0.3406874985931407, 0.46243312282827476], 
                'material': 'plasticine',
                # 'material': 'iron', 
                'center': [0.5, 0.24846874929657034, 0.5]
            }
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

    add_surface_collider(mpm_solver, (0.0, 0.6, 0.0), (0.0, -1.0, 0.0), 'slip', 1.0)

    # 防超边界
    add_bounding_box(mpm_solver)

    directory_to_save = f'./sim_results/{scene_name}'

    num_frames = 24 * 2

    head_writer = imageio.get_writer(f'./sim_results/{scene_name}.mp4', fps=24)
    left_writer = imageio.get_writer(f'./sim_results/{scene_name}_left.mp4', fps=24)
    right_writer = imageio.get_writer(f'./sim_results/{scene_name}_right.mp4', fps=24)

    scene_materials = construct_scene_material_from_materials(
        materials_range,
        materials_mapping,
        instances,
        device=dvc
    )

    # start_time = 0.
    # position = 0.25
    
    # while start_time <= 1e-5 * 1600 * 90:
    #     # print(position)
    #     start_time, position = up_and_down(
    #         mpm_solver,
    #         2.,
    #         position,
    #         0.75,
    #         start_time
    #     )

    # add_velocity_on_particles(
    #     mpm_solver,
    #     [0, 0.0, 0],
    #     start_time,
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

    cut_object(
        mpm_solver,
        bound=0.46243312282827476 + 0.25,
        v_horizon=2.336914055172607,
        v_vertical=1.2042529240319655,
        delta_t=1e-5 * 1600 * 12
    )

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
    with tqdm.tqdm(range(num_frames)) as pbar:
        for i in pbar:
            for k in range(1, 1600):
                # print(mpm_solver.mpm_state.particle_F_trial)
                J = mpm_solver.p2g2p(i, dt, device=dvc)
                J = J.numpy()
                J_lis.append(J[0])
            # print(np.any(np.isnan(mpm_solver.mpm_state.particle_x.numpy())))
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
