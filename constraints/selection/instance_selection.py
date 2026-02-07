import warp as wp
from mpm_utils import MPMStateStruct

from ..modifier import BaseModifier

@wp.kernel
def selection_particle_instance_region(
    state: MPMStateStruct,
    modifier: BaseModifier,
    target_instance: int,
    mix_type: int
):
    p = wp.tid()
    if mix_type == 1:                   # and
        condition = modifier.mask[p] == 1
    else:                               # or
        condition = True

    if (
        state.instances[p] == target_instance
        and condition
    ):
        modifier.mask[p] = 1
    else:
        modifier.mask[p] = 0