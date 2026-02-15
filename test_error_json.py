import math
import json
import time
import tqdm
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
from scene_init.materials import materials_range, materials_mapping
from scene_init.scene_render_info import construct_scene_material_from_materials

def normal_simulation_once(
    scene_info, 
    save_root, 
    idx, 
    preprocess=None,
    dvc='cuda:0',
    *args,
    **kwargs
):
    scene_name = idx
    dt = 1e-5
    position_vec, velocity_vec, volume_tensor, instances, particles = init_scene(
        scene_info
    )
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

    # 保存信息
    # directory_to_save = f'{save_root}/{scene_name}'
    directory_to_save = os.path.join(save_root, str(scene_name))
    os.makedirs(directory_to_save, exist_ok=True)

    with open(f'{save_root}/{scene_name}/scene.json', 'w') as f:
        json.dump(instances, f)

    num_frames = 24 * 5

    head_writer = imageio.get_writer(os.path.join(directory_to_save, 'movie.mp4'), fps=24)
    left_writer = imageio.get_writer(os.path.join(directory_to_save, 'left.mp4'), fps=24)
    right_writer = imageio.get_writer(os.path.join(directory_to_save, 'right.mp4'), fps=24)

    scene_materials = construct_scene_material_from_materials(
        materials_range,
        materials_mapping,
        instances,
        device=dvc
    )

    if preprocess is not None and callable(preprocess):
        preprocess(
            mpm_solver=mpm_solver,
            *args,
            **kwargs
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

    for i in tqdm.tqdm(range(num_frames), desc="Simulation", position=1, leave=False):
        for k in range(1, 1600):
            # print(mpm_solver.mpm_state.particle_F_trial)
            J = mpm_solver.p2g2p(i, dt, device=dvc)
        # print(np.any(np.isnan(mpm_solver.mpm_state.particle_x.numpy())))
        if np.any(np.isnan(mpm_solver.mpm_state.particle_x.numpy())):
            raise ValueError('get nan result')
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

    save_flow_to_file(mpm_solver, directory_to_save)
    # plt.figure(figsize=(10, 6))
    # plt.plot(frames, values, linewidth=2)
    # plt.savefig('./tmp/J_curve.png')

    # mpm_solver.renderer.clear()
    del mpm_solver
    # end = time.time()
    # print(f"use time {end - start}s")

if __name__ == "__main__":
    wp.init()
    wp.config.verify_cuda = True
    dvc = "cuda:0"
    start = time.time()
    # INIT scene
    # def add_velo(mpm_solver, *args, **kwargs):
    #     add_velocity_on_particles(
    #         mpm_solver,
    #         [0, 0, 0],
    #         0.0,
    #         999.0,
    #         mix_type=0,
    #         region_params=[
    #             {
    #                 'type': 'instance',
    #                 'param': {
    #                     'target_instance': 0
    #                 }
    #             }
    #         ],
    #         device='cuda:0'
    #     )

    normal_simulation_once(
        [
            {
                'instance_type': 'mesh', 
                'path': './assets/objs/apple.obj', 
                'cube_param': [0.2798288062133525, 0.3306641676461868, 0.16116508072076297],
                'material': 'jelly', 
                'center': [0.5072338668460352, 0.7066448183748087, 0.7646026041310372]
            }, 
            {
                'instance_type': 'mesh', 
                'path': './assets/objs/futou.obj', 
                'cube_param': [0.3490457651948291, 0.36478056918708024, 0.18207323537949027], 
                'material': 'jelly', 
                'center': [0.6662652798417561, 0.6909785525469174, 0.5049340002796018]
            }, 
            {
                'instance_type': 'cube', 
                'cube_param': [0.23277214926671652, 0.36159720690118413, 0.1694803902127917], 
                'material': 'slime', 
                'center': [0.7783876542157964, 0.6742621291508717, 0.7853561642881481]
            }, 
            {
                'instance_type': 'cube', 
                'center': (0.5, 0.2650280842358631, 0.5), 
                'cube_param': (0.9, 0.3300561684717262, 0.9), 
                'material': 'fluid'
            }
        ],
        save_root='./sim_results',
        idx=1,
        # preprocess=add_velo
    )
    end = time.time()
    print(f"use time {end - start}s")
