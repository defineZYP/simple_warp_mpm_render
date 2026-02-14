import torch
import warp as wp

from scene_init.materials import get_random_material_from_range

@wp.struct
class Scene:
    meshes: wp.array(dtype=wp.uint64)
    material_ids: wp.array(dtype=wp.int32)
    num_meshes: int

@wp.struct
class SceneMaterial:
    albedo: wp.array(dtype=wp.vec3)
    emission: wp.array(dtype=wp.vec3)
    roughness: wp.array(dtype=float)
    metallic: wp.array(dtype=float)
    transmission: wp.array(dtype=float)
    ior: wp.array(dtype=float)

def torch2warp_vec3(t, copy=False, dtype=wp.types.float32, dvc="cuda:0"):
    assert t.is_contiguous()
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    assert t.shape[1] == 3
    a = wp.types.array(
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

def torch2warp_float32(t, copy=False, dtype=wp.types.float32, dvc="cuda:0"):
    assert t.is_contiguous()
    if t.dtype != torch.float32 and t.dtype != torch.int32:
        raise RuntimeError(
            "Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type"
        )
    a = wp.types.array(
        ptr=t.data_ptr(),
        dtype=wp.float32,
        shape=t.shape[0],
        copy=False,
        # owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc,
    )
    a.tensor = t
    return a

def construct_scene_material_from_materials(materials, materials_mapping, instances, device):
    scene_material = SceneMaterial()
    num_materials = len(instances)
    albedo = torch.zeros((num_materials, 3), dtype=torch.float32, device=device)
    emission = torch.ones((num_materials, 3), dtype=torch.float32, device=device)
    roughness = torch.zeros((num_materials,), dtype=torch.float32, device=device)
    metallic = torch.zeros((num_materials,), dtype=torch.float32, device=device)
    transmission = torch.zeros((num_materials,), dtype=torch.float32, device=device)
    ior = torch.zeros((num_materials, ), dtype=torch.float32, device=device)
    for idx, instance in enumerate(instances):
        # print(instance)
        # material = get_random_material_from_range(
        #     materials[materials_mapping[instance['material']['material']]]
        # )
        material = instance['material']
        # print(material)
        if 'albedo' in material:
            albedo[idx, :] = torch.tensor(material['albedo'], dtype=torch.float32, device=device)
        if 'emission' in material:
            emission[idx, :] = torch.tensor(material['emission'], dtype=torch.float32, device=device)
        if 'roughness' in material:
            roughness[idx] = material['roughness']
        if 'metallic' in material:
            metallic[idx] = material['metallic']
        if 'transmission' in material:
            transmission[idx] = material['transmission']
        if 'ior' in material:
            ior[idx] = material['ior']
    scene_material.albedo = torch2warp_vec3(albedo, dvc=device)
    scene_material.emission = torch2warp_vec3(emission, dvc=device)
    scene_material.roughness = torch2warp_float32(roughness, dvc=device)
    scene_material.metallic = torch2warp_float32(metallic, dvc=device)
    scene_material.transmission = torch2warp_float32(transmission, dvc=device)
    scene_material.ior = torch2warp_float32(ior, dvc=device)
    # raise NotImplementedError()
    return scene_material

if __name__ == "__main__":
    from materials import materials_range
    scene_material = construct_scene_material_from_materials(
        materials_range,
        'cuda:0'
    )
    print(scene_material)
    
