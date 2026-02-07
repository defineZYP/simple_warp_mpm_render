import warp as wp
from mpm_utils import MPMStateStruct

from ..modifier import BaseModifier

@wp.kernel
def selection_particle_cube_region(
    state: MPMStateStruct, 
    modifier: BaseModifier,
    point: wp.vec3,
    size: wp.vec3,
    mix_type: int
):
    p = wp.tid()
    offset = state.particle_x[p] - point
    if mix_type == 1:                   # and
        condition = modifier.mask[p] == 1
    else:                               # or
        condition = True
    if (
        wp.abs(offset[0]) < size[0]
        and wp.abs(offset[1]) < size[1]
        and wp.abs(offset[2]) < size[2]
        and condition
    ):
        modifier.mask[p] = 1
    else:
        modifier.mask[p] = 0
