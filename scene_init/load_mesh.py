import os
import torch
import random
import math

import warp as wp
import numpy as np

from .materials import materials_range, materials_mapping, get_random_material_from_range
import trimesh

from warp_utils import torch2warp_vec3, torch2warp_float, torch2warp_int32

@wp.struct
class MeshToParticles:
    particles: wp.array(dtype=wp.vec3)
    valid: wp.array(dtype=int)

@wp.kernel
def sample_mesh_inside(
    struct: MeshToParticles,
    mesh_id: wp.uint64,
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    dy: float,
    dz: float,
    delta_x: float,
    delta_y: float,
    delta_z: float
):
    i, j, k = wp.tid()
    index = i * ny * nz + j * nz + k
    _x = (float(i) + 0.5) * dx + delta_x
    _y = (float(j) + 0.5) * dy + delta_y
    _z = (float(k) + 0.5) * dz + delta_z
    point = wp.vec3(_x, _y, _z)
    result = wp.mesh_query_point(
        mesh_id,
        point,
        10.0
    )
    # print(result.result)
    if result.result and result.sign < 0:
        struct.particles[index] = point
        struct.valid[index] = 1

    
def sample_mesh(
    obj_file: str,
    center: list,
    cube_param: list,
    cube_delta: list,
    n: int,
    device: str = 'cuda:0'
):
    m = MeshToParticles()
    m.particles = wp.zeros(
        shape=n,
        dtype=wp.vec3,
        device=device
    )
    m.valid = wp.zeros(
        shape=n,
        dtype=int,
        device=device
    )
    mesh = trimesh.load(obj_file)
    mesh.process()
    bbox_min, bbox_max = mesh.bounds
    bbox_size = bbox_max - bbox_min
    target_center = np.array(center)
    target_size = np.array(cube_param)

    scale = np.min(target_size / bbox_size) + 1e-5
    mesh.apply_scale(scale)

    bbox_min, bbox_max = mesh.bounds
    translation = target_center - (bbox_min + bbox_max) / 2

    mesh.apply_translation(translation)
    bbox_min, bbox_max = mesh.bounds
    # print(bbox_min, bbox_max)
    # print(bbox_min / 2 + bbox_max / 2)
    # mesh.export('./output.obj')

    vertices = torch2warp_vec3(
        torch.tensor(mesh.vertices, dtype=torch.float32, device=device), 
        dvc=device
    )

    indices = torch2warp_int32(
        torch.tensor(mesh.faces, dtype=torch.int32, device=device).reshape(-1),
        dvc=device
    )

    mesh = wp.Mesh(
        points=vertices,
        indices=indices
    )

    delta = target_center - (target_size / 2)
    # print(cube_delta, delta)
    wp.launch(
        kernel=sample_mesh_inside,
        dim=((cube_delta[0], cube_delta[1], cube_delta[2])),
        inputs=[m, mesh.id, cube_delta[0], cube_delta[1], cube_delta[2], cube_delta[3], cube_delta[4], cube_delta[5], delta[0], delta[1], delta[2]],
        device=device
    )

    
    valid = m.valid.numpy()
    particles = m.particles.numpy()
    particles = particles[valid == 1]
    # print(particles)

    del mesh

    return particles

def init_mesh(
    mesh_pathes=None,
    instance_params=[None],
    delta_noise_velocity=False,
    index_bias=0,
    device='cuda:0'
):
    centers = []            # mesh 的中心位置
    cube_params = []        # mesh在xyz轴的长度
    cube_deltas = []
    materials = []
    num_particles = []
    instances = []
    velocities = []

    # 两轮，第一轮采样基本参数，第二轮获取最终结果
    for i_idx in range(len(mesh_pathes)):
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
            velocity = [random.uniform(-0.1, 0.1), random.uniform(0.05, 0.1), random.uniform(-0.05, 0.05)]
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
    
    total_particles = int(np.sum(num_particles)) + 1000 * len(mesh_pathes)
    position_vec = torch.zeros((total_particles, 3), dtype=torch.float32)
    velocity_vec = torch.zeros((total_particles, 3), dtype=torch.float32)
    volumn_vec = torch.zeros((total_particles), dtype=torch.float32)
    start_particle_idx = 0
    # 第二轮
    for i_idx in range(len(mesh_pathes)):
        _position_vec = sample_mesh(
            mesh_pathes[i_idx],
            centers[i_idx],
            cube_params[i_idx],
            cube_deltas[i_idx],
            num_particles[i_idx],
            device=device
        )
        true_num_particles = _position_vec.shape[0]
        end_particle_idx = start_particle_idx + true_num_particles
        position_vec[start_particle_idx: end_particle_idx] = torch.tensor(_position_vec)
        velocity_vec[start_particle_idx: end_particle_idx, :] = torch.tensor([velocities[i_idx]]).repeat(true_num_particles, 1)
        if delta_noise_velocity:
            velocity_vec[start_particle_idx: end_particle_idx, :] += torch.rand((true_num_particles, 3)) * 1e-2
        volumn_vec[start_particle_idx: end_particle_idx] = torch.ones(true_num_particles, dtype=torch.float32) / materials[i_idx]['particle_dense']
        instances.append({
            'start_idx': start_particle_idx + index_bias,
            'end_idx': end_particle_idx + index_bias,
            'material': materials[i_idx],
        })
        start_particle_idx = end_particle_idx

    return position_vec[:end_particle_idx], velocity_vec[:end_particle_idx], volumn_vec[:end_particle_idx], instances, centers, cube_params, end_particle_idx

    
if __name__ == "__main__":
    wp.init()

    density = 100000
    cube_param = [1, 1, 1]
    lx_dense = np.pow(density, 1/3) 
    nx = int(cube_param[0] * lx_dense)
    ny = int(cube_param[1] * lx_dense)
    nz = int(cube_param[2] * lx_dense)
    dx = cube_param[0] / nx
    dy = cube_param[1] / ny
    dz = cube_param[2] / nz
    cube_delta = [nx, ny, nz, dx, dy, dz]
    num_particles = nx * ny * nz

    mesh = sample_mesh(
        '/DATA/DATANAS1/zhangyip/phy/warp-mpm/assets/objs/dragon.obj',
        center=[0, 0, 0],
        cube_param=cube_param,
        cube_delta=cube_delta,
        n=num_particles
    )

