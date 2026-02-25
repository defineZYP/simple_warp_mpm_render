import torch

from .random_ball import init_ball
from .random_cube import init_cube
from .load_mesh import init_mesh
from .random_nerd_cube import init_nerd_cube

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
    total_particles = 0
    position_vec = []
    velocity_vec = []
    volumn_vec = []
    instances = []

    # merge same instances
    for instance_param in instance_params:
        assert 'instance_type' in instance_param
        assert instance_param is not None
        instance_type = instance_param['instance_type']
        if instance_type == "ball":
            _position_vec, _velocity_vec, _volumn_vec, _instances, _, _, _particles = init_ball(
                1,
                [instance_param],
                index_bias=total_particles
            )
        elif instance_type == "cube":
            _position_vec, _velocity_vec, _volumn_vec, _instances, _, _, _particles = init_cube(
                1,
                [instance_param],
                index_bias=total_particles
            )
        elif instance_type == "mesh":
            # mesh_instances.append(instance_param)
            # mesh_pathes.append(instance_param['path'])
            _position_vec, _velocity_vec, _volumn_vec, _instances, _, _, _particles = init_mesh(
                [instance_param['path']],
                [instance_param],
                index_bias=total_particles,
                device=device
            )
        elif instance_type == "nerd_cube":
            _position_vec, _velocity_vec, _volumn_vec, _instances, _, _, _particles = init_nerd_cube(
                1,
                [instance_param],
                index_bias=total_particles
            )
        else:
            raise NotImplementedError(f'Instance type {instance_type} has not supported yet.')
        
        position_vec.append(_position_vec)
        velocity_vec.append(_velocity_vec)
        volumn_vec.append(_volumn_vec)
        instances += _instances
        total_particles += _particles
    
    # init balls
    # ball_position_vec, ball_velocity_vec, ball_volumn_vec, ball_instances, _, _, ball_particles = init_ball(
    #     len(ball_instances),
    #     ball_instances
    # )

    # print(ball_particles)

    # init cubes
    # cube_position_vec, cube_velocity_vec, cube_volumn_vec, cube_instances, _, _, cube_particles = init_cube(
    #     len(cube_instances),
    #     cube_instances,
    #     index_bias=ball_particles
    # )

    # print(cube_particles)
    
    # init mesh
    # mesh_position_vec, mesh_velocity_vec, mesh_volumn_vec, mesh_instances, _, _, mesh_particles = init_mesh(
    #     mesh_pathes,
    #     mesh_instances,
    #     index_bias=ball_particles + cube_particles,
    #     device=device
    # )

    # print(mesh_particles)

    # merge
    # total_particles = ball_particles + cube_particles + mesh_particles
    position_vec = torch.cat(position_vec, dim=0)
    velocity_vec = torch.cat(velocity_vec, dim=0)
    volumn_vec = torch.cat(volumn_vec, dim=0)
    # instances = ball_instances + cube_instances + mesh_instances
    
    return position_vec, velocity_vec, volumn_vec, instances, total_particles
    