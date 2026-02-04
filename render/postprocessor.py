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
        wp.pow(rgb[0], gamma),
        wp.pow(rgb[1], gamma),
        wp.pow(rgb[2], gamma)
    )
    return rgb