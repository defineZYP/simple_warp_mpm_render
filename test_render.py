import os
import torch

import numpy as np
import warp as wp

import trimesh

from render.renderer import PathTracingRender
from render.camera import Camera, init_camera
from scene_init.scene_render_info import Scene, SceneMaterial, construct_scene_material_from_materials
from scene_init.materials import materials_range

def load_obj_as_warp_mesh(path, device="cuda"):
    # 加载 obj
    mesh = trimesh.load(path, force='mesh')

    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("Loaded object is not a single mesh")

    # vertices: (N, 3)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)

    # faces: (M, 3)
    indices = np.asarray(mesh.faces, dtype=np.int32).reshape(-1)

    # 转 Warp array
    wp_vertices = wp.array(vertices, dtype=wp.vec3, device=device)
    wp_indices  = wp.array(indices, dtype=wp.int32, device=device)

    # 构建 Warp Mesh（自动建 BVH）
    warp_mesh = wp.Mesh(
        points=wp_vertices,
        indices=wp_indices,
    )

    return warp_mesh

if __name__ == "__main__":
    import time
    obj_file = '/DATA/DATANAS1/zhangyip/phy/learn_nvisll/content/dragon.obj'
    start = time.time()
    mesh = load_obj_as_warp_mesh(obj_file)
    # print(mesh.points)
    points = mesh.points.numpy()
    print(points.shape)
    print(np.mean(points, axis=0))
    print(np.max(points, axis=0))
    print(np.min(points, axis=0))
    camera = init_camera(
        position=(-0.42515832, 4.203418, -0.09355669 + 15),
        lookat=(-0.42515832, 4.203418, -0.09355669),
        up=(0.0, 1.0, 0.0),
        fov=45,
        width=512,
        height=512,
        exposure=1.0,
        aperture=0.0,
        focus_distance=0.2,
        near=0.1,
        gamma=2.2
    )
    renderer = PathTracingRender(
        cameras=[camera],
        sample_per_pixel=16
    )
    scene_materials = construct_scene_material_from_materials(materials=materials_range, device='cuda:0')
    meshes = torch.zeros((1,), dtype=torch.uint64, device='cuda:0')
    meshes[0] = mesh.id
    _meshes = wp.types.array(
        ptr=meshes.data_ptr(),
        dtype=wp.uint64,
        shape=meshes.shape[0],
        copy=False,
        # owner=False,
        requires_grad=meshes.requires_grad,
        # device=t.device.type)
        device='cuda:0',
    )
    _meshes.tensor = meshes

    material_ids = torch.zeros((1,), dtype=torch.int32, device='cuda:0')
    material_ids[0] = 0
    _material_ids = wp.types.array(
        ptr=material_ids.data_ptr(),
        dtype=wp.int32,
        shape=material_ids.shape[0],
        copy=False,
        # owner=False,
        requires_grad=material_ids.requires_grad,
        # device=t.device.type)
        device='cuda:0',
    )
    _material_ids.tensor = material_ids

    scene = Scene()
    scene.meshes = _meshes
    scene.material_ids = _material_ids
    scene.num_meshes = 1
    
    for i in range(10):
        img = renderer.render(
            camera_id=0,
            scene=scene,
            scene_materials=scene_materials
        )
    end = time.time()
    print(f"{end - start} s", img.shape)

    from PIL import Image

    img = np.clip(img, 0.0, 1.0).reshape(512, 512, 3)
    img_u8 = (img * 255.0).astype(np.uint8)

    Image.fromarray(img_u8, mode="RGB").save("output.png")