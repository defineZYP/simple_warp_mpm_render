import torch

import numpy as np
import warp as wp

from .ray import Ray
from .camera import Camera, init_camera, generate_ray
from .scene import hdr_texture, HDRImage

from scene_init.scene_render_info import Scene, SceneMaterial

light_quality = 512.0
ENV_IOR = 1.000277
VISIBILITY   = 0.000001

@wp.func
def vec3_mul(x: wp.vec3, y: wp.vec3):
    return wp.vec3(x[0] * y[0], x[1] * y[1], x[2] * y[2])

@wp.func
def vec3_div(x: wp.vec3, y: wp.vec3):
    return wp.vec3(x[0] / y[0], x[1] / y[1], x[2] / y[2])

@wp.func
def sample_spherical_map(v: wp.vec3):
    v = wp.normalize(v)
    phi = wp.atan2(v.z, v.x)
    theta = wp.asin(v.y)
    _u = 0.5 + phi * (0.5 / wp.pi)
    _v = 0.5 - theta * (1.0 / wp.pi)

    return wp.vec2(_u, _v)

@wp.func
def brightness(rgb: wp.vec3) -> float:
    return wp.dot(rgb, wp.vec3(0.299, 0.587, 0.114))

@wp.func
def linear_mix(
    x: wp.vec3,
    y: wp.vec3,
    a: float
):
    return x * (1.0 - a) + y * a

@wp.func
def float_mix(
    x: float,
    y: float,
    a: float
):
    return x * (1.0 - a) + y * a

@wp.func
def hemispheric_sampling(
    n: wp.vec3,
    tid: int
):
    z = 2.0 * wp.randf(wp.uint32(tid + 42)) - 1.0
    a = wp.randf(wp.uint32(tid + 43)) * 2.0 * wp.pi

    xy = wp.sqrt(1.0 - z * z) * wp.vec2(wp.sin(a), wp.cos(a))

    normal = n + wp.vec3(xy[0], xy[1], z)
    return wp.normalize(normal)

@wp.func
def sample_cdf(
    cdf: wp.array(dtype=float),
    xi: float
):
    lo = int(0)
    hi = int(cdf.shape[0] - 1)
    while lo < hi:
        mid = (lo + hi) // 2
        if xi <= cdf[mid]:
            hi = mid
        else:
            lo = mid + 1
    # if lo >= cdf.shape[0]:
    #     print(99999)
    return lo

@wp.func
def hdr_importance_hemispheric_sampling(
    hdr_cdf: wp.array(dtype=float),
    height: int,
    width: int,
    n: wp.vec3,
    tid: int
):
    xi = wp.randf(wp.uint32(tid + 42))
    idx = sample_cdf(hdr_cdf, xi)
    x = float(idx % width)
    y = float(idx // width)
    u = (x + 0.5) / float(width)
    v = (y + 0.5) / float(height)
    phi = 2.0 * wp.pi * u
    theta = wp.pi * v

    sin_theta = wp.sin(theta)
    wi = wp.vec3(
        wp.cos(phi) * sin_theta,
        wp.cos(theta),
        wp.sin(phi) * sin_theta
    )
    if wp.dot(wi, n) < 0.0:
        wi = -wi

    return wp.normalize(wi)

@wp.func
def roughness_sampling(
    hemispheric_sample: wp.vec3,
    n: wp.vec3,
    roughness: float,
):
    alpha = roughness * roughness
    rough_normal = linear_mix(
        n, hemispheric_sample, alpha
    )
    return wp.normalize(rough_normal)

@wp.func
def fresnel_schlick(
    NoI: float,
    F0: float, 
    roughness: float
):
    return float_mix(float_mix(wp.pow(abs(1.0 + NoI), 5.0), 1.0, F0), F0, roughness)

@wp.func
def calculate_sky_color(
    hdr: HDRImage,
    direction: wp.vec3,
):
    uv = sample_spherical_map(direction)
    color = hdr_texture(
        hdr, uv
    )
    return color, uv

@wp.func
def ray_surface_interaction(
    hdr_image: HDRImage,
    materials: SceneMaterial,
    material_id: int,
    ray: Ray,
    hit_n: wp.vec3,
    hit_point: wp.vec3,
    hit_sign: float,
    hdr_cdf: wp.array(dtype=float),
    height: int,
    width: int,
    depth: int,
    x: float,
    y: float,
    tid: int
):
    albedo = materials.albedo[material_id]
    roughness = materials.roughness[material_id]
    metallic = materials.metallic[material_id]
    transmission = materials.transmission[material_id]
    ior = materials.ior[material_id]

    outer = hit_sign > 0
    if outer:
        normal = hit_n
    else:
        normal = -1.0 * hit_n
    # raise NotImplementedError()

    # hemispheric_sample = hemispheric_sampling(normal, tid)

    hemispheric_sample = hdr_importance_hemispheric_sampling(
        hdr_cdf,
        height,
        width,
        normal,
        tid
    )

    roughness_sample = roughness_sampling(
        hemispheric_sample,
        normal,
        roughness
    )

    N   = roughness_sample
    I   = ray.direction
    NoI = wp.dot(N, I)

    eta = ENV_IOR / ior if outer else ior / ENV_IOR
    k   = 1.0 - eta * eta * (1.0 - NoI * NoI)
    F0  = (eta - 1.0) / (eta + 1.0); F0 *= 2.0*F0

    F = fresnel_schlick(NoI, F0, roughness)

    if wp.randf(wp.uint32(tid + 520)) < F + metallic or k < 0.0:
        ray.direction = I - 2.0 * NoI * N
        ray.color *= float(wp.dot(ray.direction, normal) > 0.0)
        # wp.printf(ray.color)
        # ray.color = vec3_mul(ray.color, float(wp.dot(ray.direction, normal) > 0.0))
    elif wp.randf(wp.uint32(tid + 521)) < transmission:
        ray.direction = eta * I - (wp.sqrt(k) + eta * NoI) * N
    else:
        ray.direction = hemispheric_sample
    
    ray.color = vec3_mul(ray.color, albedo)
    
    ray.origin = hit_point

    return ray

@wp.kernel
def render_frame(
    camera: Camera,
    hdr_image: HDRImage,
    scene: Scene,
    materials: SceneMaterial,
    frame_buffer: wp.array(dtype=wp.vec4)
):
    tid = wp.tid()
    width = int(camera.width + 0.5)
    num_pixels = camera.num_pixels
    xy = tid % num_pixels
    x = float(xy % width)
    y = float(xy // width)

    # init ray
    ray = generate_ray(camera, x, y, tid)

    for i in range(512):
        if i >= 3:
            inv_pdf = wp.exp(float(i) / light_quality)
            roulette_prob = 1.0 - (1.0 / inv_pdf)
            rand_value = wp.randf(wp.uint32(tid + 23))
            if rand_value < roulette_prob:
                ray.color /= roulette_prob
                break
        t_min = float(1000.0)
        hit_n = wp.vec3(0.0, 0.0, 0.0)
        hit_mesh_idx = int(-1)
        hit_sign = float(-1.0)
        for mesh_idx in range(scene.num_meshes):
            mesh = scene.meshes[mesh_idx]
            query = wp.mesh_query_ray(
                mesh,
                ray.origin,
                ray.direction,
                100.0
            )
            if query.result and query.t < t_min and query.t > 1e-5:
                # if hit
                t_min = query.t
                hit_n = query.normal
                hit_sign = query.sign
                hit_mesh_idx = mesh_idx
        # if x == 985 and y == 395:
        #     print(hit_mesh_idx)
        if hit_mesh_idx == -1:
            sky_color, uv = calculate_sky_color(hdr_image, ray.direction)
            ray.color = vec3_mul(ray.color, sky_color)
            break
        # or hit a mesh, ray surface interaction
        hit_point = ray.origin + ray.direction * t_min
        # if hit_mesh_idx >= scene.material_ids.shape[0]:
        #     print(114514)
        material_id = scene.material_ids[hit_mesh_idx]

        ray = ray_surface_interaction(
            hdr_image=hdr_image,
            materials=materials,
            material_id=material_id,
            ray=ray,
            hit_n=hit_n,
            hit_point=hit_point,
            hit_sign=hit_sign,
            hdr_cdf=hdr_image.cdf,
            height=hdr_image.height,
            width=hdr_image.width,
            depth=i,
            x=x,
            y=y,
            tid=tid
        )

        intensity = brightness(ray.color)
        # ray.color *= materials.emission[material_id]
        ray.color += vec3_mul(ray.color, materials.emission[material_id])
        visible = brightness(ray.color)
        if intensity + 1e-5 < visible or visible < VISIBILITY:
            break

    wp.atomic_add(
        frame_buffer, xy, wp.vec4(ray.color[0], ray.color[1], ray.color[2], 1.0)
    )
    