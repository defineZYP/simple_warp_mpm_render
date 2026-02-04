import numpy as np
import warp as wp

@wp.struct
class Ray:
    origin: wp.vec3
    direction: wp.vec3
    color: wp.vec3
    depth: int
    