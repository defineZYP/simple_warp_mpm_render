# a unit modifier
import warp as wp

@wp.struct
class BaseModifier:
    ### region of the modifier
    # point: wp.vec3                      # center          
    # size: wp.vec3                       # cube
    # normal: wp.vec3                     # cylinder normal
    # half_height_and_radius: wp.vec2     # cylinder 

    mask: wp.array(dtype=int)             # change which point

    # time
    start_time: float
    end_time: float

    # rotation information
    rotation_scale: float
    translation_scale: float
    point: wp.vec3
    normal: wp.vec3
    direction: wp.vec3
    horizontal_axis_1: wp.vec3
    horizontal_axis_2: wp.vec3

    # velocity information
    velocity: wp.vec3
    velocityTimesDt: wp.vec3

    # force
    force: wp.vec3
    forceTimesDt: wp.vec3
    numsteps: int
    