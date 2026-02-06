import warp as wp

# from render import vec3_mul

@wp.func
def adjust(
    rgb: wp.vec3,
    exposure: float,
    gamma: float
):
    rgb = rgb * exposure
    rgb = wp.vec3(
        wp.clamp(wp.pow(rgb[0], 1.0 / gamma), 0.0, 1.0),
        wp.clamp(wp.pow(rgb[1], 1.0 / gamma), 0.0, 1.0),
        wp.clamp(wp.pow(rgb[2], 1.0 / gamma), 0.0, 1.0)
    )
    return rgb