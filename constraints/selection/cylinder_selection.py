import warp as wp
from mpm_utils import MPMStateStruct

from ..modifier import BaseModifier

@wp.kernel
def selection_particle_cylinder_region(
    state: MPMStateStruct,
    modifier: BaseModifier,
    point: wp.vec3,
    normal: wp.vec3,
    half_height_and_radius: wp.vec2,
    mix_type: int
):
    p = wp.tid()
    offset = state.particle_x[p] - point

    vertical_distance = wp.abs(wp.dot(offset, normal))
    horizontal_distance = wp.length(
        offset - wp.dot(offset, normal) * normal
    )
    if mix_type == 1:                   # and
        condition = modifier.mask[p] == 1
    else:                               # or
        condition = True
    if (
        vertical_distance < half_height_and_radius[0]
        and horizontal_distance < half_height_and_radius[1]
        and condition
    ):
        modifier.mask[p] = 1
    else:
        modifier.mask[p] = 0
