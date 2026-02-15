import torch

from .random_ball import init_ball
from .random_cube import init_cube
from .load_mesh import init_mesh

def init_scene(
    instance_params=[None],
    delta_noise_velocity=False,
    device='cuda:0'
):
    assert instance_params is not None
    assert isinstance(instance_params, list)
    assert len(instance_params) > 0
    # 目前只支持球、方块
    # TODO Mesh
    ball_instances = []
    cube_instances = []
    mesh_instances = []
    mesh_pathes = []

    # merge same instances
    for instance_param in instance_params:
        assert 'instance_type' in instance_param
        assert instance_param is not None
        instance_type = instance_param['instance_type']
        if instance_type == "ball":
            ball_instances.append(instance_param)
        elif instance_type == "cube":
            cube_instances.append(instance_param)
        elif instance_type == "mesh":
            mesh_instances.append(instance_param)
            mesh_pathes.append(instance_param['path'])
        else:
            raise NotImplementedError(f'Instance type {instance_type} has not supported yet.')
    
    # init balls
    ball_position_vec, ball_velocity_vec, ball_volumn_vec, ball_instances, _, _, ball_particles = init_ball(
        len(ball_instances),
        ball_instances
    )

    # print(ball_particles)

    # init cubes
    cube_position_vec, cube_velocity_vec, cube_volumn_vec, cube_instances, _, _, cube_particles = init_cube(
        len(cube_instances),
        cube_instances,
        index_bias=ball_particles
    )

    # print(cube_particles)
    
    # init mesh
    mesh_position_vec, mesh_velocity_vec, mesh_volumn_vec, mesh_instances, _, _, mesh_particles = init_mesh(
        mesh_pathes,
        mesh_instances,
        index_bias=ball_particles + cube_particles,
        device=device
    )

    # print(mesh_particles)

    # merge
    total_particles = ball_particles + cube_particles + mesh_particles
    position_vec = torch.cat([ball_position_vec, cube_position_vec, mesh_position_vec], dim=0)
    velocity_vec = torch.cat([ball_velocity_vec, cube_velocity_vec, mesh_velocity_vec], dim=0)
    volumn_vec = torch.cat([ball_volumn_vec, cube_volumn_vec, mesh_volumn_vec], dim=0)
    instances = ball_instances + cube_instances + mesh_instances
    
    return position_vec, velocity_vec, volumn_vec, instances, total_particles
    