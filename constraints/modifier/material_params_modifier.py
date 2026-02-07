import warp as wp

# TODO this class is not useful now, must reimplement it...

@wp.struct
class MaterialParamsModifier:
    point: wp.vec3
    size: wp.vec3
    E: float
    nu: float
    density: float