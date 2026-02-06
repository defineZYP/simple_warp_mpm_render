import torch

from .random_ball import init_ball
from .random_cube import init_cube

def init_scene(
    instance_params=[None],
    delta_noise_velocity=False,
):
    assert instance_params is not None
    assert isinstance(instance_params, list)
    assert len(instance_params) > 0
    # 目前只支持球、方块
    # TODO Mesh
    ball_instances = []
    cube_instances = []

    # merge same instances
    for instance_param in instance_params:
        assert 'instance_type' in instance_param
        assert instance_param is not None
        instance_type = instance_param['instance_type']
        if instance_type == "ball":
            ball_instances.append(instance_param)
        elif instance_type == "cube":
            cube_instances.append(instance_param)
        else:
            raise NotImplementedError(f'Instance type {instance_type} has not supported yet.')
    
    # init balls
    ball_position_vec, ball_velocity_vec, ball_volumn_vec, ball_instances, _, _, ball_particles = init_ball(
        len(ball_instances),
        ball_instances
    )

    # init cubes
    cube_position_vec, cube_velocity_vec, cube_volumn_vec, cube_instances, _, _, cube_particles = init_cube(
        len(cube_instances),
        cube_instances,
        index_bias=ball_particles
    )
    
    # merge
    total_particles = ball_particles + cube_particles
    position_vec = torch.cat([ball_position_vec, cube_position_vec], dim=0)
    velocity_vec = torch.cat([ball_velocity_vec, cube_velocity_vec], dim=0)
    volumn_vec = torch.cat([ball_volumn_vec, cube_volumn_vec], dim=0)
    instances = ball_instances + cube_instances
    
    return position_vec, velocity_vec, volumn_vec, instances, total_particles
    