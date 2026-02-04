import numpy as np
import h5py
import os
import sys
import warp as wp
import torch

import trimesh

from skimage import img_as_ubyte
from scene_init.materials import materials_mapping, materials_range
from scene_init.scene_render_info import Scene
from warp_utils import torch2warp_int32, torch2warp_uint64


def save_data_at_frame(
        mpm_solver, 
        dir_name, 
        frame, 
        save_to_ply = True, 
        save_to_h5 = False, 
        save_to_video=True, 
        video_writer=None,
        instances=None,
        scene_materials=None,
        device='cuda:0'
    ):
    os.umask(0)
    os.makedirs(dir_name, 0o777, exist_ok=True)
    
    fullfilename = dir_name + '/sim_' + str(frame).zfill(10) + '.h5'

    if save_to_ply:
        particle_position_to_ply(mpm_solver, fullfilename[:-2]+'ply')
    
    if save_to_h5:

        if os.path.exists(fullfilename): os.remove(fullfilename)
        newFile = h5py.File(fullfilename, "w")

        x_np = mpm_solver.mpm_state.particle_x.numpy().transpose() # x_np has shape (3, n_particles)
        newFile.create_dataset("x", data=x_np) # position

        currentTime = np.array([mpm_solver.time]).reshape(1,1)
        newFile.create_dataset("time", data=currentTime) # current time

        f_tensor_np = mpm_solver.mpm_state.particle_F.numpy().reshape(-1,9).transpose() # shape = (9, n_particles)
        newFile.create_dataset("f_tensor", data=f_tensor_np) # deformation grad

        v_np = mpm_solver.mpm_state.particle_v.numpy().transpose() # v_np has shape (3, n_particles)
        newFile.create_dataset("v", data=v_np) # particle velocity

        C_np = mpm_solver.mpm_state.particle_C.numpy().reshape(-1,9).transpose() # shape = (9, n_particles)
        newFile.create_dataset("C", data=C_np) # particle C
        print("save siumlation data at frame ", frame, " to ", fullfilename)

    if save_to_video:
        add_frame(mpm_solver, video_writer, instances, scene_materials, device)

def add_light(position, device='cuda:0'):
    # init a light at (0,0,0)
    v0 = torch.tensor([ 0.01767767,  0.01767767,  0.01767767], device=device) + position
    v1 = torch.tensor([ 0.01767767, -0.01767767, -0.01767767], device=device) + position
    v2 = torch.tensor([-0.01767767,  0.01767767, -0.01767767], device=device) + position
    v3 = torch.tensor([-0.01767767, -0.01767767,  0.01767767], device=device) + position
    f0 = torch.tensor([0, 1, 2], device=device)
    f1 = torch.tensor([0, 3, 1], device=device)
    f2 = torch.tensor([0, 2, 3], device=device)
    f3 = torch.tensor([1, 3, 2], device=device)
    n0 = torch.tensor([ 0.57735027,  0.57735027,  0.57735027], device=device)
    n1 = torch.tensor([ 0.57735027, -0.57735027, -0.57735027], device=device)
    n2 = torch.tensor([-0.57735027,  0.57735027, -0.57735027], device=device)
    n3 = torch.tensor([-0.57735027, -0.57735027,  0.57735027], device=device)

    verts = torch.stack([v0, v1, v2, v3], dim=0)
    indices = torch.stack([f0, f1, f2, f3], dim=0)
    vn = torch.stack([n0, n1, n2, n3], dim=0)

    texture = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 1.0, 5000.0, 5000.0, 5000.0])
    texture = np.repeat(texture[None, :], 4, axis=0)
    size = 32
    mask = np.zeros((size, size), dtype=bool)
    x, y = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    mask[x + y <= size] = 1

    texture = np.repeat(np.repeat(texture[..., None], size, axis=-1)[..., None], size, axis=-1)
    texture[:, :, ~mask] = 0.0
    texture = torch.tensor(texture, device=device)

    triangles = verts[indices]
    vn = vn[indices]
    return triangles.to(torch.float32), vn.to(torch.float32), texture.to(torch.float32)

@wp.kernel
def compute_vertex_normals(
    verts: wp.array(dtype=wp.vec3f),
    indices: wp.array(dtype=wp.int32),
    normals: wp.array(dtype=wp.vec3f)
):
    face_id = wp.tid()
    i0 = indices[face_id * 3]
    i1 = indices[face_id * 3 + 1]
    i2 = indices[face_id * 3 + 2]

    v0 = verts[i0]
    v1 = verts[i1]
    v2 = verts[i2]

    face_normal = wp.cross(v1 - v0, v2 - v0)

    wp.atomic_add(normals, i0, face_normal)
    wp.atomic_add(normals, i1, face_normal)
    wp.atomic_add(normals, i2, face_normal)

def transfer_mesh(verts, indices, vn, instance, device='cuda:0'):
    verts = torch.tensor(verts, device=device).reshape(-1, 3)
    indices = torch.tensor(indices, device=device).reshape(-1, 3)
    vn = torch.tensor(vn, device=device).reshape(-1, 3)

    size = 32
    mask = np.zeros((size, size), dtype=bool)
    x, y = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    mask[x + y <= size] = 1
    mask = torch.tensor(mask, device=device)
    
    vn = torch.nn.functional.normalize(vn, dim=1)
    triangles = verts[indices]
    vn = verts[indices]

    material = materials_range[materials_mapping[instance['material']['material']]]
    diffuse = torch.tensor(material['albedo'], device=device).unsqueeze(0).repeat(triangles.shape[0], 1)
    specular = torch.tensor([0., 0., 0.], device=device).unsqueeze(0).repeat(triangles.shape[0], 1)
    roughness = torch.tensor([material['roughness']], device=device).unsqueeze(0).repeat(triangles.shape[0], 1)
    normal = torch.zeros_like(diffuse)
    normal[..., 0] = 0.5
    normal[..., 1] = 0.5
    normal[..., 2] = 1.
    irradiance = torch.tensor(material['emission'], device=device).unsqueeze(0).repeat(triangles.shape[0], 1)
    texture = torch.cat([diffuse, specular, roughness, normal, irradiance], dim=1)
    # texture = torch.repeat(torch.repeat(texture[..., None], size, axis=-1)[..., None], size, dim=-1)
    texture = texture.unsqueeze(-1).unsqueeze(-1)
    texture = texture.repeat(1, 1, size, size)
    
    texture[:, :, ~mask] = 0.0
    return triangles.to(torch.float32), vn.to(torch.float32), texture.to(torch.float32)  

def add_frame(mpm_solver, video_writer, instances=None, scene_materials=None, device='cuda:0'):
    channels = 3
    grid_m = mpm_solver.mpm_state.grid_m_instances
    # print(position)

    # TODO mesh and color

    all_triangles = []
    all_vn = []
    all_texture = []

    if instances is not None and len(instances) > 0:
        for i_idx, instance in enumerate(instances):
            with wp.ScopedTimer(
                f'Surface Extraction {i_idx}',
                synchronize=True,
                print=False,
            ):
                mpm_solver.MCs[i_idx].surface(
                    grid_m[i_idx], 
                    0.2
                )
                vn = wp.zeros(shape=mpm_solver.MCs[i_idx].verts.shape, dtype=wp.vec3f, device=device)
                wp.launch(
                    kernel=compute_vertex_normals,
                    dim=mpm_solver.MCs[i_idx].indices.shape[0] // 3,
                    inputs=[mpm_solver.MCs[i_idx].verts, mpm_solver.MCs[i_idx].indices],
                    outputs=[vn],
                    device=device
                )
                triangles, vn, texture = transfer_mesh(
                    verts=mpm_solver.MCs[i_idx].verts.numpy(),
                    indices=mpm_solver.MCs[i_idx].indices.numpy(),
                    vn=vn.numpy(),
                    instance=instance,
                    device=device
                )
                all_triangles.append(triangles)
                all_vn.append(vn)
                all_texture.append(texture)
    
    triangles, vn, texture = add_light(
        torch.tensor([0.0, 2.0, 0.0], device=device),
        device=device
    )
    all_triangles.append(triangles)
    all_vn.append(vn)
    all_texture.append(texture)
    
    all_triangles = torch.cat(all_triangles, dim=0).unsqueeze(0)
    all_vn = torch.cat(all_vn, dim=0).unsqueeze(0)
    all_texture = torch.cat(all_texture, dim=0).unsqueeze(0)

    mask = torch.ones((1, all_triangles.shape[1]), dtype=torch.bool, device=device)

    # print(all_triangles.dtype)
    # print(all_texture.dtype)
    # print(mask.dtype)
    # print(all_vn.dtype)
    # print(mpm_solver.c2w.dtype)
    # print(mpm_solver.fov.dtype)

    rendered_imgs = mpm_solver.renderer(
        triangles=all_triangles,
        texture=all_texture,
        mask=mask,
        vn=all_vn,
        c2w=mpm_solver.c2w,
        fov=mpm_solver.fov,
        resolution=512,
        torch_dtype=torch.float16,
    )
    rendered_imgs = rendered_imgs[0, 0].cpu().numpy().astype(np.float32)
    pixels_np = np.clip(rendered_imgs, 0., 1.)

    # pixels_np = np.clip(pixels_np, 0.0, 1.0)

    # pixels_np = pixels_np.numpy()
    video_writer.append_data(img_as_ubyte(pixels_np))


def particle_position_to_ply(mpm_solver, filename):
    # position is (n,3)
    if os.path.exists(filename):
        os.remove(filename)
    position = mpm_solver.mpm_state.particle_x.numpy()
    num_particles = (position).shape[0]
    position = position.astype(np.float32)
    # print(position.shape)
    with open(filename, 'wb') as f: # write binary
        header = f"""ply
            format binary_little_endian 1.0
            element vertex {num_particles}
            property float x
            property float y
            property float z
            end_header
        """
        f.write(str.encode(header))
        f.write(position.tobytes())
        print("write", filename)

def particle_position_tensor_to_ply(position_tensor, filename):
    # position is (n,3)
    if os.path.exists(filename):
        os.remove(filename)
    position = position_tensor.clone().detach().cpu().numpy()
    num_particles = (position).shape[0]
    position = position.astype(np.float32)
    with open(filename, 'wb') as f: # write binary
        header = f"""ply
            format binary_little_endian 1.0
            element vertex {num_particles}
            property float x
            property float y
            property float z
            end_header
        """
        f.write(str.encode(header))
        f.write(position.tobytes())
        print("write", filename)
    
if __name__ == "__main__":
    triangles, vn, texture = add_light(torch.zeros((3,), device='cuda:0'), device='cuda:0')
    print(triangles.shape, vn.shape, texture.shape)
    print(triangles[0, 1, :])