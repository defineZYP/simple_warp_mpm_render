import warp as wp

from .base_modifier import BaseModifier

class Impulse_modifier:
    def __init__(
        self,
        point: wp.vec3,
        size: wp.vec3,
        normal: wp.vec3,
        modifier: BaseModifier
    ):
        self.point = point
        self.size = size
        self.normal = normal
        self.modifier = modifier