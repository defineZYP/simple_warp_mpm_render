import warp as wp

@wp.struct
class ParticleForceModifier:
    point: wp.vec3
    normal: wp.vec3
    half_height_and_radius: wp.vec2
    rotation_scale: float
    translation_scale: float

    size: wp.vec3

    horizontal_axis_1: wp.vec3
    horizontal_axis_2: wp.vec3
    
    start_time: float

    end_time: float

    velocity: wp.vec3

    mask: wp.array(dtype=int)