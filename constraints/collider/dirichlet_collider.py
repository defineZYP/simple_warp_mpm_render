import warp as wp

# for various boundary conditions
@wp.struct
class Dirichlet_collider:
    point: wp.vec3
    normal: wp.vec3
    direction: wp.vec3

    start_time: float
    end_time: float

    friction: float
    surface_type: int

    velocity: wp.vec3

    threshold: float
    reset: int
    index: int

    x_unit: wp.vec3
    y_unit: wp.vec3
    radius: float
    v_scale: float
    width: float
    height: float
    length: float
    R: float

    size: wp.vec3

    horizontal_axis_1: wp.vec3
    horizontal_axis_2: wp.vec3
    half_height_and_radius: wp.vec2