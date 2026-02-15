import numpy as np
import h5py
import os
import sys
import warp as wp
import torch

from skimage import img_as_ubyte
from scene_init.materials import materials_mapping
from scene_init.scene_render_info import Scene
from warp_utils import torch2warp_int32, torch2warp_uint64, MPMStateStruct


def save_data_at_frame(
        mpm_solver, 
        dir_name, 
        frame, 
        save_to_ply = True, 
        save_to_h5 = False, 
        save_to_video=True, 
        save_to_flow=False,
        dt=1e-5,
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

    if save_to_flow:
        save_optical_flow(mpm_solver, dt, device)

def save_optical_flow(mpm_solver, dt, device):
    mpm_solver.optical_renderer.render(
        camera_id=0,
        n_particles=mpm_solver.n_particles,
        mpm_state=mpm_solver.mpm_state,
        dt=dt,
        device=device
    )

def save_flow_to_file(mpm_solver, dir_name):
    depth = np.stack(mpm_solver.optical_renderer.depths).astype(np.float32)
    force = np.stack(mpm_solver.optical_renderer.forces).astype(np.float32)
    flow = np.stack(mpm_solver.optical_renderer.flows).astype(np.float32)
    np.savez_compressed(os.path.join(dir_name, "depth.npz"), depth)
    np.savez_compressed(os.path.join(dir_name, "force.npz"), force)
    np.savez_compressed(os.path.join(dir_name, "flow.npz"), flow)

def add_frame(mpm_solver, video_writers, instances=None, scene_materials=None, device='cuda:0'):
    channels = 3
    grid_m = mpm_solver.mpm_state.grid_m_instances
    # print(position)
    # TODO mesh and color
    meshes = []
    meshes_ids = torch.zeros((len(instances) + 1, ), dtype=torch.uint64, device=device)
    material_ids = torch.zeros((len(instances) + 1, ), dtype=torch.int32, device=device)

    if instances is not None and len(instances) > 0:
        for i_idx, instance in enumerate(instances):
            with wp.ScopedTimer(
                f'Surface Extraction {i_idx}',
                synchronize=True,
                print=False,
            ):
                mpm_solver.MCs[i_idx].surface(
                    grid_m[i_idx], 
                    0.1
                )
                # print(type(grid_m[i_idx]))
                arr = grid_m[i_idx].numpy()
                # print(i_idx, arr[arr != 0].mean(), arr[arr != 0].max(), arr[arr != 0].min())
                # print(mpm_solver.MCs[i_idx].indices.numpy().shape)
                mesh = wp.Mesh(
                    points=mpm_solver.MCs[i_idx].verts,
                    indices=mpm_solver.MCs[i_idx].indices
                )
                meshes.append(
                    mesh
                )
                meshes_ids[i_idx] = mesh.id
                # material_ids[i_idx] = materials_mapping[instance['material']['material']]
                material_ids[i_idx] = i_idx
    
    meshes_ids = torch2warp_uint64(meshes_ids)
    material_ids = torch2warp_int32(material_ids)

    scene = Scene()
    scene.meshes = meshes_ids
    scene.material_ids = material_ids
    # scene.num_meshes = len(instances) + 1
    scene.num_meshes = len(instances)
    
    for i in range(len(mpm_solver.renderer.cameras)):
        video_writer = video_writers[i]
        pixels_np = mpm_solver.renderer.render(
            camera_id=i,
            scene=scene,
            scene_materials=scene_materials
        )

        pixels_np = np.clip(pixels_np, 0.0, 1.0).reshape(
            int(mpm_solver.renderer.cameras[i].height), 
            int(mpm_solver.renderer.cameras[i].width), 
            3
        )

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
    