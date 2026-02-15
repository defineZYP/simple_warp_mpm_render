import os
import torch
import random
import math

import numpy as np

from .materials import materials_range, materials_mapping, get_random_material_from_range

def sample_ball(
    center,
    radius,
    cube_delta,
    n,
):
    # n_guess = int(n / 0.5)
    # pts = np.random.uniform(-radius, radius, (n_guess, 3))
    # dist_sq = np.sum(pts ** 2, axis=1)
    # inside = pts[dist_sq <= radius ** 2]
    # if len(inside) < n:
    #     inside += np.array(center)
    #     return np.concatenate((inside, sample_ball(center, radius, n - len(inside))), axis=0)
    # else:
    #     return inside[:n] + np.array(center)
    pts = np.zeros((n, 3), dtype=np.float32)
    nx, dx = cube_delta
    for i in range(nx):
        for j in range(nx):
            for k in range(nx):
                index = i * nx * nx + j * nx + k
                pts[index, 0] = (i + 0.5) * dx - radius 
                pts[index, 1] = (j + 0.5) * dx - radius 
                pts[index, 2] = (k + 0.5) * dx - radius 
    dist_sq = np.sum(pts ** 2, axis=1)
    inside = pts[dist_sq <= radius ** 2] + np.array(center)
    return inside, len(inside)

def init_ball(
    num_instances=1,
    instance_params=[None],
    delta_noise_velocity=False,
    index_bias=0
):
    """
    random init a scene of balls
    num_instances: number of instances
    instance_params: list of dict, define the physical parameters of instances
    delta_noise_velocity: whether to give a small noise to the velocities of particles
    """
    centers = []
    radiuses = []
    cube_deltas = []
    materials = []
    num_particles = []
    instances = []
    velocities = []

    # 两轮，第一轮采样基本参数，第二轮获取最终结果
    for i_idx in range(num_instances):
        # init instance params
        radius = -1
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
            if 'radius' in instance_param:
                radius = instance_param['radius']
            if 'material' in instance_param:
                material = instance_param['material']
            if 'center' in instance_param:
                center = instance_param['center']
            if 'velocity' in instance_param:
                velocity = instance_param['velocity']
        if radius == -1:
            radius = random.uniform(0.05, 0.5)
        radiuses.append(radius)

        if center == -1:
            if radius >= 0.5:
                center = [0.5, 0.5, 0.5]
            else:
                center = [random.uniform(0.1 + radius, 0.9 - radius), random.uniform(0.1 + radius, 0.9 - radius), random.uniform(0.1 + radius, 0.9 - radius)]
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
        
        if material['material'] == "foam":
            # to avoid too less particles
            material['particle_dense'] = material['density'] * 10000
        else:
            material['particle_dense'] = min(1000000.0, material['density'] * 1000)
        materials.append(material)

        # sample particles to simulate the ball
        lx_dense = np.pow(material['particle_dense'], 1/3) 
        nx = int(2 * radius * lx_dense)
        dx = 2 * radius / nx
        cube_deltas.append([nx, dx])
        num_particle = nx ** 3 + 1
        num_particles.append(num_particle)

    total_particles = int(np.sum(num_particles))
    total_particles = int(total_particles * 0.53) + 2000 * num_instances
    position_vec = torch.zeros((total_particles, 3), dtype=torch.float32)
    velocity_vec = torch.zeros((total_particles, 3), dtype=torch.float32)
    volumn_vec = torch.zeros((total_particles), dtype=torch.float32)
    start_particle_idx = 0
    end_particle_idx = 0
    # 第二轮
    for i_idx in range(num_instances):
        _position_vec, true_num_particles = sample_ball(
            centers[i_idx],
            radiuses[i_idx],
            cube_deltas[i_idx],
            num_particles[i_idx]
        )
        end_particle_idx = start_particle_idx + true_num_particles
        # print(position_vec.shape)
        # print(start_particle_idx, end_particle_idx, true_num_particles, _position_vec.shape)
        position_vec[start_particle_idx: end_particle_idx] = torch.tensor(_position_vec)
        velocity_vec[start_particle_idx: end_particle_idx, :] = torch.tensor([velocities[i_idx]]).repeat(true_num_particles, 1)
        # print(velocities[i_idx])
        if delta_noise_velocity:
            velocity_vec[start_particle_idx: end_particle_idx, :] += torch.rand((true_num_particles, 3)) * 1e-2
        volumn_vec[start_particle_idx: end_particle_idx] = torch.ones(true_num_particles, dtype=torch.float32) / materials[i_idx]['particle_dense']
        instances.append({
            'start_idx': start_particle_idx + index_bias,
            'end_idx': end_particle_idx + index_bias,
            'material': materials[i_idx],
            'velocity': velocities[i_idx],
            'center': centers[i_idx],
            'radiuses': radiuses[i_idx]
        })
        start_particle_idx = end_particle_idx

    return position_vec[:end_particle_idx], velocity_vec[:end_particle_idx], volumn_vec[:end_particle_idx], instances, centers, radiuses, end_particle_idx
