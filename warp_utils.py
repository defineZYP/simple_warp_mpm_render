import warp as wp
import warp.torch
import torch
import numpy as np

@wp.struct
class MPMModelStruct:
    ####### essential #######
    grid_lim: float
    n_particles: int
    n_grid: int
    dx: float
    inv_dx: float
    grid_dim_x: int
    grid_dim_y: int
    grid_dim_z: int
    mu: wp.array(dtype=float)
    lam: wp.array(dtype=float)
    E: wp.array(dtype=float)
    nu: wp.array(dtype=float)
    bulk: wp.array(dtype=float)
    material: int

    ######## for plasticity ####
    yield_stress: wp.array(dtype=float)
    friction_angle: wp.array(dtype=float)
    alpha: wp.array(dtype=float)
    xi: wp.array(dtype=float)
    hardening: wp.array(dtype=float)
    softening: wp.array(dtype=float)
    plastic_viscosity: wp.array(dtype=float)
    gravitational_accelaration: wp.vec3
    

    ####### for damping
    grid_v_damping_scale: float
    rpic_damping: wp.array(dtype=float)

    ####### for PhysGaussian: covariance
    update_cov_with_F: int


@wp.struct
class MPMStateStruct:
    ###### essential #####
    # instances
    instances: wp.array(dtype=int)
    materials: wp.array(dtype=int)
    # particle
    particle_x: wp.array(dtype=wp.vec3)   # current position
    particle_v: wp.array(dtype=wp.vec3)   # particle velocity
    particle_tf: wp.array(dtype=wp.vec3)   # particle total force
    particle_F: wp.array(dtype=wp.mat33)  # particle elastic deformation gradient
    particle_init_cov: wp.array(dtype=float)  # initial covariance matrix
    particle_cov: wp.array(dtype=float)     # current covariance matrix
    particle_F_trial: wp.array(
        dtype=wp.mat33
    )  # apply return mapping on this to obtain elastic def grad
    particle_R: wp.array(dtype=wp.mat33)  # rotation matrix
    particle_stress: wp.array(dtype=wp.mat33)  # Kirchoff stress, elastic stress
    particle_C: wp.array(dtype=wp.mat33)
    particle_vol: wp.array(dtype=float)  # current volume
    particle_mass: wp.array(dtype=float)  # mass
    particle_density: wp.array(dtype=float)  # density
    particle_Jp: wp.array(dtype=float)

    particle_selection: wp.array(dtype=int) # only particle_selection[p] = 0 will be simulated

    # grid
    grid_m: wp.array(dtype=float, ndim=3)
    grid_m_instances: wp.array(dtype=float, ndim=4)
    # grid_m_instances_render: wp.array(dtype=float, ndim=4)
    grid_v_in: wp.array(dtype=wp.vec3, ndim=3)  # grid node momentum/velocity
    grid_v_out: wp.array(
        dtype=wp.vec3, ndim=3
    )  # grid node momentum/velocity, after grid update

@wp.struct
class MPMtailoredStruct:
    # this needs to be changed for each different BC!
    point: wp.vec3
    normal: wp.vec3
    start_time: float
    end_time: float
    friction: float
    surface_type: int
    velocity: wp.vec3
    threshold: float
    reset: int

    point_rotate: wp.vec3
    normal_rotate: wp.vec3
    x_unit: wp.vec3
    y_unit: wp.vec3
    radius: float
    v_scale: float
    width: float
    point_plane: wp.vec3
    normal_plane: wp.vec3
    velocity_plane: wp.vec3
    threshold_plane: float

@wp.kernel
def set_vec3_to_zero(target_array: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    target_array[tid] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def set_mat33_to_identity(target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@wp.kernel
def add_identity_to_mat33(target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.add(
        target_array[tid], wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    )


@wp.kernel
def subtract_identity_to_mat33(target_array: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.sub(
        target_array[tid], wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    )


@wp.kernel
def add_vec3_to_vec3(
    first_array: wp.array(dtype=wp.vec3), second_array: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    first_array[tid] = wp.add(first_array[tid], second_array[tid])


@wp.kernel
def set_value_to_float_array(target_array: wp.array(dtype=float), value: float):
    tid = wp.tid()
    target_array[tid] = value

@wp.kernel
def set_value_to_int_array(target_array: wp.array(dtype=int), value: int):
    tid = wp.tid()
    target_array[tid] = value

@wp.kernel
def get_float_array_product(
    arrayA: wp.array(dtype=float),
    arrayB: wp.array(dtype=float),
    arrayC: wp.array(dtype=float),
):
    tid = wp.tid()
    arrayC[tid] = arrayA[tid] * arrayB[tid]

@wp.kernel
def set_friction_angle_and_alpha(
    friction_angle: wp.array(dtype=float),
    alpha: wp.array(dtype=float),
    value: float
):
    tid = wp.tid()
    friction_angle[tid] = value
    sin_phi = wp.sin(value / 180.0 * 3.14159265)
    alpha[tid] = wp.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)

def torch2warp_uint64(t, copy=False, dvc='cuda:0'):
    assert t.is_contiguous()
    if t.dtype != torch.uint64:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be uint64 type"
        )
    a = wp.types.array(
        ptr=t.data_ptr(),
        dtype=wp.uint64,
        shape=t.shape[0],
        copy=False,
        # owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a

def torch2warp_int32(t, copy=False, dvc='cuda:0'):
    assert t.is_contiguous()
    if t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be int32 type"
        )
    a = wp.types.array(
        ptr=t.data_ptr(),
        dtype=wp.int32,
        shape=t.shape[0],
        copy=False,
        # owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a

def torch2warp_quat(t, copy=False, dtype=warp.types.float32, dvc="cuda:0"):
    assert t.is_contiguous()
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    assert t.shape[1] == 4
    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=wp.quat,
        shape=t.shape[0],
        copy=False,
        owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a

def torch2warp_float(t, copy=False, dtype=warp.types.float32, dvc="cuda:0"):
    assert t.is_contiguous()
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=warp.types.float32,
        shape=t.shape[0],
        copy=False,
        # owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a

def torch2warp_vec2(t, copy=False, dtype=warp.types.float32, dvc="cuda:0"):
    assert t.is_contiguous()
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    assert t.shape[1] == 2
    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=wp.vec2,
        shape=t.shape[0],
        copy=False,
        # owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a

def torch2warp_vec3(t, copy=False, dtype=warp.types.float32, dvc="cuda:0"):
    assert t.is_contiguous()
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    assert t.shape[1] == 3
    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=wp.vec3,
        shape=t.shape[0],
        copy=False,
        # owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a

def torch2warp_mat33(t, copy=False, dtype=warp.types.float32, dvc="cuda:0"):
    assert t.is_contiguous()
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    assert t.shape[1] == 3
    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=wp.mat33,
        shape=t.shape[0],
        copy=False,
        owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a