import warp as wp
from .base_modifier import BaseModifier

def init_modifier(mix_type, n_particles, device):
    modifier = BaseModifier()
    if mix_type == 1:
        # and 
        modifier.mask = wp.ones(
            shape=n_particles,
            dtype=int,
            device=device
        )
    else:
        modifier.mask = wp.zeros(
            shape=n_particles,
            dtype=int,
            device=device
        )
    return modifier