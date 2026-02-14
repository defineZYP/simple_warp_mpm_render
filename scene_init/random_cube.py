import os
import torch
import random
import math

import numpy as np

from .materials import materials_range, materials_mapping, get_random_material_from_range

def sample_cube(
    center,
    cube_param,
    cube_delta,
    n
):
    pts = np.zeros((n, 3), dtype=np.float32)
    nx, ny, nz, dx, dy, dz = cube_delta
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                index = i * ny * nz + j * nz + k
                pts[index, 0] = (i + 0.5) * dx - cube_param[0] / 2 + center[0]
                pts[index, 1] = (j + 0.5) * dy - cube_param[1] / 2 + center[1]
                pts[index, 2] = (k + 0.5) * dz - cube_param[2] / 2 + center[2]
    return pts

def init_cube(
    num_instances=1,
    instance_params=[None],
    delta_noise_velocity=False,
    index_bias=0
):
    """
    random init a scene of balls
    num_instances: number of instances
    particle_per_instance: number of particles to simulate an instance
    instance_params: list of dict, define the physical parameters of instances
    """
    centers = []                     # 立方体的中心位置
    cube_params = []                 # 立方体的在xyz轴的长度，每个元素以(x, y, z)出现
    cube_deltas = []                 # 记录delta
    materials = []                   # 材料
    num_particles = []               # 粒子数量
    instances = []                   # 物体整体性质
    velocities = []                 # 物体初始化的速度

    # 两轮，第一轮采样基本参数，第二轮获取最终结果
    for i_idx in range(num_instances):
        # init instance params
        cube_param = -1
        material = -1
        center = -1
        velocity = -1
        if (
            instance_params is not None and 
            isinstance(instance_params, list) and 
            len(instance_params) > i_idx and 
            instance_params[i_idx] is not None and 
            isinstance(instance_params[i_idx], dict)
        ):
            # 给定了实体的一些参数
            instance_param = instance_params[i_idx]
            if 'cube_param' in instance_param:
                cube_param = instance_param['cube_param']
            if 'material' in instance_param:
                material = instance_param['material']
            if 'center' in instance_param:
                center = instance_param['center']
            if 'velocity' in instance_param:
                velocity = instance_param['velocity']

        if cube_param == -1:
            cube_param = [random.uniform(0.01, 0.05), random.uniform(0.01, 0.05), random.uniform(0.01, 0.05)]
        cube_params.append(cube_param)

        if center == -1:
            radius = np.max(cube_param)
            if radius >= 0.5:
                center = [0.5, 0.5, 0.5]
            else:
                center = [random.uniform(0.3 + radius, 0.7 - radius), random.uniform(0.3 + radius, 0.7 - radius), random.uniform(0.3 + radius, 0.7 - radius)]
        centers.append(center)

        if velocity == -1:
            velocity = [random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2)]
        velocities.append(velocity)

        if material == -1:
            # TODO random choose a material and randomly change the params
            material_type = random.randint(0, len(materials_range) - 1)
            material_range = materials_range[material_type]
            material = get_random_material_from_range(material_range)
        if isinstance(material, str):
            material_range = materials_range[materials_mapping[material]]
            material = get_random_material_from_range(material_range)
        
        material['particle_dense'] = material['density'] * 1000
        materials.append(material)
        
        # sample particles to simulate the ball
        lx_dense = np.pow(material['particle_dense'], 1/3) 
        nx = int(cube_param[0] * lx_dense)
        ny = int(cube_param[1] * lx_dense)
        nz = int(cube_param[2] * lx_dense)
        dx = cube_param[0] / nx
        dy = cube_param[1] / ny
        dz = cube_param[2] / nz
        cube_deltas.append([nx, ny, nz, dx, dy, dz])
        num_particle = nx * ny * nz
        num_particles.append(num_particle)

    total_particles = int(np.sum(num_particles))
    position_vec = torch.zeros((total_particles, 3), dtype=torch.float32)
    velocity_vec = torch.zeros((total_particles, 3), dtype=torch.float32)
    volumn_vec = torch.zeros((total_particles), dtype=torch.float32)
    start_particle_idx = 0
    # 第二轮
    for i_idx in range(num_instances):
        end_particle_idx = start_particle_idx + num_particles[i_idx]
        position_vec[start_particle_idx: end_particle_idx, :] = torch.tensor(
            sample_cube(
                centers[i_idx],
                cube_params[i_idx],
                cube_deltas[i_idx],
                num_particles[i_idx]
            )
        )
        velocity_vec[start_particle_idx: end_particle_idx, :] = torch.tensor([velocities[i_idx]]).repeat(num_particles[i_idx], 1)
        if delta_noise_velocity:
            velocity_vec[start_particle_idx: end_particle_idx, :] += torch.rand((num_particles[i_idx], 3)) * 1e-2
        volumn_vec[start_particle_idx: end_particle_idx] = torch.ones(num_particles[i_idx], dtype=torch.float32) / materials[i_idx]['particle_dense']
        instances.append({
            'start_idx': start_particle_idx + index_bias,
            'end_idx': end_particle_idx + index_bias,
            'material': materials[i_idx],
            'velocity': velocities[i_idx],
            'center': centers[i_idx],
            'cube_param': cube_params[i_idx]
        })
        start_particle_idx = end_particle_idx

    return position_vec, velocity_vec, volumn_vec, instances, centers, cube_params, total_particles
