import sys
import os
import random
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from engine_utils import *
from warp_utils import *
from mpm_utils import *
import warp.render

# from renderformer_utils import look_at_to_c2w

from render.renderer import PathTracingRender
from render.camera import Camera, init_camera
from render.optical_flow_renderer import OpticalFlowRenderer
# from scene_init.scene_render_info import Scene, SceneMaterial, construct_scene_material_from_materials
# from scene_init.materials import materials_range

import pyglet
pyglet.options["headless"] = True

from renderformer import RenderFormerRenderingPipeline

c_res = wp.zeros(shape=1, dtype=float)

@wp.kernel
def get_J(state: MPMStateStruct, c_res: wp.array(dtype=float)):
    tid = wp.tid()
    F = state.particle_F[tid]
    c_res[tid] = wp.determinant(F)
    # print(type(c_res))

class MPM_Simulator_WARP:
    def __init__(
        self, 
        n_particles, 
        n_grid=100, 
        grid_lim=1.0, 
        num_instances=1, 
        rgb_renderer=None,
        optical_renderer=None,
        device="cuda:0"
    ):
        self.init_times = 0
        self.num_instances = num_instances
        
        self.renderer = rgb_renderer

        self.optical_renderer = optical_renderer

        self.initialize(n_particles, n_grid, grid_lim, num_instances=num_instances, device=device)
        self.time_profile = {}
        self.instances = []
        self.colors = []
        self.MCs = []

    def initialize(self, n_particles, n_grid=100, grid_lim=1.0, num_instances=1, device="cuda:0"):
        self.n_particles = n_particles
        self.num_instances = num_instances
        self.n_grid = n_grid
        print(f"initialize {num_instances}")
        # init marching cubes
        self.MCs = [
            wp.MarchingCubes(
                n_grid, 
                n_grid, 
                n_grid, 
                1e5, 
                1e5,
                domain_bounds_lower_corner=(0.0, 0.0, 0.0),
                domain_bounds_upper_corner=(1.0, 1.0, 1.0),
            ) for _ in range(num_instances)
        ]
        self.mpm_model = MPMModelStruct()
        self.mpm_model.n_particles = n_particles
        # domain will be [0,grid_lim]*[0,grid_lim]*[0,grid_lim] !!!
        # domain will be [0,grid_lim]*[0,grid_lim]*[0,grid_lim] !!!
        # domain will be [0,grid_lim]*[0,grid_lim]*[0,grid_lim] !!!
        self.mpm_model.grid_lim = grid_lim
        self.mpm_model.n_grid = n_grid
        self.mpm_model.grid_dim_x = self.mpm_model.n_grid
        self.mpm_model.grid_dim_y = self.mpm_model.n_grid
        self.mpm_model.grid_dim_z = self.mpm_model.n_grid
        (
            self.mpm_model.dx,
            self.mpm_model.inv_dx,
        ) = self.mpm_model.grid_lim / self.mpm_model.n_grid, float(
            self.mpm_model.n_grid / self.mpm_model.grid_lim
        )

        self.mpm_model.E = wp.zeros(shape=n_particles, dtype=float, device=device)
        self.mpm_model.nu = wp.zeros(shape=n_particles, dtype=float, device=device)
        self.mpm_model.mu = wp.zeros(shape=n_particles, dtype=float, device=device)
        self.mpm_model.lam = wp.zeros(shape=n_particles, dtype=float, device=device)
        # 新版没有这个玩意儿
        self.mpm_model.bulk = wp.zeros(shape=n_particles, dtype=float, device=device)
        self.mpm_model.update_cov_with_F = False


        # material is used to switch between different elastoplastic models. 0 is jelly
        # prepare materials
        self.mpm_model.material = 0

        self.mpm_model.plastic_viscosity = wp.zeros(
            shape=num_instances, dtype=float, device=device
        )
        # self.mpm_model.softening = 0.1

        self.mpm_model.yield_stress = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )
        
        self.mpm_model.friction_angle = wp.zeros(
            shape=num_instances, dtype=float, device=device
        )
        self.mpm_model.alpha = wp.zeros(
            shape=num_instances, dtype=float, device=device
        )
        
        wp.launch(
            kernel=set_friction_angle_and_alpha,
            dim=num_instances,
            inputs=[self.mpm_model.friction_angle, self.mpm_model.alpha, 25.0],
            device=device
        )

        self.mpm_model.xi = wp.zeros(
            shape=num_instances, dtype=float, device=device
        )

        self.mpm_model.hardening = wp.zeros(
            shape=num_instances, dtype=float, device=device
        )

        self.mpm_model.softening = wp.zeros(
            shape=num_instances, dtype=float, device=device
        )
        wp.launch(
            kernel=set_value_to_float_array,
            dim=num_instances,
            inputs=[self.mpm_model.softening, 0.1],
            device=device
        )

        self.mpm_model.gravitational_accelaration = wp.vec3(0.0, 0.0, 0.0)

        # self.mpm_model.rpic_damping = 0.0  # 0.0 if no damping (apic). -1 if pic
        self.mpm_model.rpic_damping = wp.zeros(
            shape=num_instances, dtype=float, device=device
        )

        self.mpm_model.grid_v_damping_scale = 1.1  # globally applied

        self.mpm_state = MPMStateStruct()

        self.mpm_state.materials = wp.zeros(shape=num_instances, dtype=int, device=device)
        self.mpm_state.instances = wp.zeros(shape=n_particles, dtype=int, device=device)

        self.mpm_state.particle_x = wp.empty(
            shape=n_particles, dtype=wp.vec3, device=device
        )  # current position

        self.mpm_state.particle_v = wp.zeros(
            shape=n_particles, dtype=wp.vec3, device=device
        )  # particle velocity

        self.mpm_state.particle_tf = wp.zeros(
            shape=n_particles, dtype=wp.vec3, device=device
        )  # particle velocity

        self.mpm_state.particle_F = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )  # particle F elastic

        self.mpm_state.particle_R = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )  # particle R rotation

        self.mpm_state.particle_init_cov = wp.zeros(
            shape=n_particles * 6, dtype=float, device=device
        )  # initial covariance matrix

        self.mpm_state.particle_cov = wp.zeros(
            shape=n_particles * 6, dtype=float, device=device
        )  # current covariance matrix

        self.mpm_state.particle_F_trial = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )  # apply return mapping will yield

        self.mpm_state.particle_stress = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )

        self.mpm_state.particle_vol = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )  # particle volume
        self.mpm_state.particle_mass = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )  # particle mass
        self.mpm_state.particle_density = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )
        self.mpm_state.particle_C = wp.zeros(
            shape=n_particles, dtype=wp.mat33, device=device
        )
        self.mpm_state.particle_Jp = wp.zeros(
            shape=n_particles, dtype=float, device=device
        )

        self.mpm_state.particle_selection = wp.zeros(
            shape=n_particles, dtype=int, device=device
        )

        self.mpm_state.grid_m = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=float,
            device=device,
        )

        self.mpm_state.grid_m_instances = wp.zeros(
            shape=(self.num_instances, self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=float,
            device=device
        )

        self.mpm_state.grid_v_in = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )
        self.mpm_state.grid_v_out = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )

        self.time = 0.0

        self.grid_postprocess = []
        self.collider_params = []
        self.modify_bc = []

        self.tailored_struct_for_bc = MPMtailoredStruct()
        self.pre_p2g_operations = []
        self.impulse_params = []

        self.particle_velocity_modifiers = []
        self.particle_velocity_modifier_params = []

    # the h5 file should store particle initial position and volume.
    def load_from_sampling(
        self, sampling_h5, n_grid=100, grid_lim=1.0, device="cuda:0"
    ):
        if not os.path.exists(sampling_h5):
            print("h5 file cannot be found at ", os.getcwd() + sampling_h5)
            exit()

        h5file = h5py.File(sampling_h5, "r")
        x, particle_volume = h5file["x"], h5file["particle_volume"]

        x = x[()].transpose()  # np vector of x # shape now is (n_particles, dim)

        self.dim, self.n_particles = x.shape[1], x.shape[0]

        self.initialize(self.n_particles, n_grid, grid_lim, device=device)

        print(
            "Sampling particles are loaded from h5 file. Simulator is re-initialized for the correct n_particles"
        )
        particle_volume = np.squeeze(particle_volume, 0)

        self.mpm_state.particle_x = wp.from_numpy(
            x, dtype=wp.vec3, device=device
        )  # initialize warp array from np

        # initial velocity is default to zero
        wp.launch(
            kernel=set_vec3_to_zero,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_v],
            device=device,
        )
        # initial velocity is default to zero

        # initial deformation gradient is set to identity
        wp.launch(
            kernel=set_mat33_to_identity,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_F_trial],
            device=device,
        )
        # initial deformation gradient is set to identity

        self.mpm_state.particle_vol = wp.from_numpy(
            particle_volume, dtype=float, device=device
        )

        print("Particles initialized from sampling file.")
        print("Total particles: ", self.n_particles)

    # shape of tensor_x is (n, 3); shape of tensor_volume is (n,)
    def load_initial_data_from_torch(
        self,
        tensor_x,
        tensor_v,
        tensor_volume,
        tensor_cov = None,
        n_grid=100,
        grid_lim=1.0,
        num_instances=1,
        device="cuda:0",
    ):
        self.dim, self.n_particles = tensor_x.shape[1], tensor_x.shape[0]
        assert tensor_x.shape[0] == tensor_volume.shape[0]
        # assert tensor_x.shape[0] == tensor_cov.reshape(-1, 6).shape[0]
        self.initialize(self.n_particles, n_grid, grid_lim, num_instances=num_instances, device=device)

        self.import_particle_x_from_torch(tensor_x, device)
        self.mpm_state.particle_vol = wp.from_numpy(
            tensor_volume.detach().clone().cpu().numpy(), dtype=float, device=device
        )
        if tensor_cov is not None:
            self.mpm_state.particle_init_cov = wp.from_numpy(
                tensor_cov.reshape(-1).detach().clone().cpu().numpy(),
                dtype=float,
                device=device,
            )

            if self.mpm_model.update_cov_with_F:
                self.mpm_state.particle_cov = self.mpm_state.particle_init_cov

        # initial velocity is default to zero
        if tensor_v is None:
            wp.launch(
                kernel=set_vec3_to_zero,
                dim=self.n_particles,
                inputs=[self.mpm_state.particle_v],
                device=device,
            )
        else:
            self.import_particle_v_from_torch(tensor_v, device)
            # print(f"init velocity {self.mpm_state.particle_v.numpy()}")
        # initial velocity is default to zero

        # initial deformation gradient is set to identity
        wp.launch(
            kernel=set_mat33_to_identity,
            dim=self.n_particles,
            inputs=[self.mpm_state.particle_F_trial],
            device=device,
        )
        # initial trial deformation gradient is set to identity

        print("Particles initialized from torch data.")
        print("Total particles: ", self.n_particles)

    # must give density. mass will be updated as density * volume
    def set_parameters(self, device="cuda:0", **kwargs):
        self.set_parameters_dict(device, kwargs)

    def set_parameters_instances(self, global_kwargs={}, instances=[], device='cuda:0'):
        # global kwargs
        if "grid_lim" in global_kwargs:
            self.mpm_model.grid_lim = global_kwargs["grid_lim"]
        if "n_grid" in global_kwargs:
            self.mpm_model.n_grid = global_kwargs["n_grid"]
        self.mpm_model.grid_dim_x = self.mpm_model.n_grid
        self.mpm_model.grid_dim_y = self.mpm_model.n_grid
        self.mpm_model.grid_dim_z = self.mpm_model.n_grid
        (
            self.mpm_model.dx,
            self.mpm_model.inv_dx,
        ) = self.mpm_model.grid_lim / self.mpm_model.n_grid, float(
            self.mpm_model.n_grid / self.mpm_model.grid_lim
        )
        self.mpm_state.grid_m = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=float,
            device=device,
        )
        self.mpm_state.grid_v_in = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )
        self.mpm_state.grid_v_out = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )

        if "grid_v_damping_scale" in global_kwargs:
            self.mpm_model.grid_v_damping_scale = global_kwargs["grid_v_damping_scale"]

        if "g" in global_kwargs:
            self.mpm_model.gravitational_accelaration = wp.vec3(global_kwargs["g"][0], global_kwargs["g"][1], global_kwargs["g"][2])

        # material, E, nu, bulk_modulus, yield_stress, density
        is_density = False
        for i_idx, instance in enumerate(instances):
            start_idx = instance['start_idx']
            end_idx = instance['end_idx']
            material = instance['material']
            material_type = material['material']
            if material_type == 'jelly':
                material_type = 0
            elif material_type == "metal":
                material_type = 1
            elif material_type == "sand":
                material_type = 2
            elif material_type == "foam":
                material_type = 3
            elif material_type == "snow":
                material_type = 4
            elif material_type == "plasticine":
                material_type = 5
            elif material_type == "fluid":
                material_type = 6
            else:
                raise TypeError("Undefined material type")

            wp.launch(
                kernel=set_value_to_int_array,
                dim=end_idx - start_idx,
                inputs=[self.mpm_state.instances[start_idx: end_idx], i_idx],
                device=device,
            )
            # print(len(self.mpm_state.materials))
            wp.launch(
                kernel=set_value_to_int_array,
                dim=1,
                inputs=[self.mpm_state.materials[i_idx: i_idx + 1], material_type],
                device=device,
            )

            if "E" in material:
                wp.launch(
                    kernel=set_value_to_float_array,
                    dim=end_idx - start_idx,
                    inputs=[self.mpm_model.E[start_idx: end_idx], material["E"]],
                    device=device,
                )
            if "nu" in material:
                wp.launch(
                    kernel=set_value_to_float_array,
                    dim=end_idx - start_idx,
                    inputs=[self.mpm_model.nu[start_idx: end_idx], material["nu"]],
                    device=device,
                )
            if "bulk_modulus" in material:
                wp.launch(
                    kernel=set_value_to_float_array,
                    dim=end_idx - start_idx,
                    inputs=[self.mpm_model.bulk[start_idx: end_idx], material["bulk_modulus"]],
                    device=device
                )
            if "yield_stress" in material:
                val = material["yield_stress"]
                wp.launch(
                    kernel=set_value_to_float_array,
                    dim=end_idx - start_idx,
                    inputs=[self.mpm_model.yield_stress[start_idx: end_idx], val],
                    device=device,
                )
            if "friction_angle" in material:
                val = material["friction_angle"]
                wp.launch(
                    kernel=set_friction_angle_and_alpha,
                    dim=1,
                    inputs=[self.mpm_model.friction_angle[i_idx: i_idx + 1], self.mpm_model.alpha[i_idx: i_idx + 1], val],
                    device=device
                )
            if "xi" in material:
                val = material["xi"]
                wp.launch(
                    kernel=set_value_to_float_array,
                    dim=1,
                    inputs=[self.mpm_model.xi[i_idx: i_idx + 1], val],
                    device=device
                )
            if "hardening" in material:
                val = material["hardening"]
                # self.mpm_model.hardening
                wp.launch(
                    kernel=set_value_to_float_array,
                    dim=1,
                    inputs=[self.mpm_model.hardening[i_idx: i_idx + 1], val],
                    device=device
                )
            if "softening" in material:
                val = material['softening']
                wp.launch(
                    kernel=set_value_to_float_array,
                    dim=1,
                    inputs=[self.mpm_model.softening[i_idx: i_idx + 1], val],
                    device=device
                )
            if "rpic_damping" in material:
                val = material['rpic_damping']
                wp.launch(
                    kernel=set_value_to_float_array,
                    dim=1,
                    inputs=[self.mpm_model.rpic_damping[i_idx: i_idx + 1], val],
                    device=device
                )
            if "plastic_viscosity" in material:
                val = material['plastic_viscosity']
                wp.launch(
                    kernel=set_value_to_float_array,
                    dim=1,
                    inputs=[self.mpm_model.plastic_viscosity[i_idx: i_idx + 1], val],
                    device=device
                )
            if "density" in material:
                density = material['density']
                wp.launch(
                    kernel=set_value_to_float_array,
                    dim=end_idx - start_idx,
                    inputs=[self.mpm_state.particle_density[start_idx: end_idx], density],
                    device=device
                )
                is_density = True 
        if is_density:
            wp.launch(
                kernel=get_float_array_product,
                dim=self.n_particles,
                inputs=[
                    self.mpm_state.particle_density,
                    self.mpm_state.particle_vol,
                    self.mpm_state.particle_mass,
                ],
                device=device,
            )
        self.num_instances = len(instances)

    def set_parameters_dict(self, kwargs={}, device="cuda:0"):
        if "material" in kwargs:
            if kwargs["material"] == "jelly":
                # self.mpm_model.material = 0
                wp.launch(
                    kernel=set_value_to_int_array,
                    dim=self.n_particles,
                    inputs=[self.mpm_state.materials, 0],
                    device=device,
                )
            elif kwargs["material"] == "metal":
                # self.mpm_model.material = 1
                wp.launch(
                    kernel=set_value_to_int_array,
                    dim=self.n_particles,
                    inputs=[self.mpm_state.materials, 1],
                    device=device,
                )
            elif kwargs["material"] == "sand":
                # self.mpm_model.material = 2
                wp.launch(
                    kernel=set_value_to_int_array,
                    dim=self.n_particles,
                    inputs=[self.mpm_state.materials, 2],
                    device=device,
                )
            elif kwargs["material"] == "foam":
                # self.mpm_model.material = 3
                wp.launch(
                    kernel=set_value_to_int_array,
                    dim=self.n_particles,
                    inputs=[self.mpm_state.materials, 3],
                    device=device,
                )
            elif kwargs["material"] == "snow":
                # self.mpm_model.material = 4
                wp.launch(
                    kernel=set_value_to_int_array,
                    dim=self.n_particles,
                    inputs=[self.mpm_state.materials, 4],
                    device=device,
                )
            elif kwargs["material"] == "plasticine":
                # self.mpm_model.material = 5
                wp.launch(
                    kernel=set_value_to_int_array,
                    dim=self.n_particles,
                    inputs=[self.mpm_state.materials, 5],
                    device=device,
                )
            elif kwargs["material"] == "fluid":
                # self.mpm_model.material = 6
                wp.launch(
                    kernel=set_value_to_int_array,
                    dim=self.n_particles,
                    inputs=[self.mpm_state.materials, 6],
                    device=device,
                )
            else:
                raise TypeError("Undefined material type")

        if "grid_lim" in kwargs:
            self.mpm_model.grid_lim = kwargs["grid_lim"]
        if "n_grid" in kwargs:
            self.mpm_model.n_grid = kwargs["n_grid"]
        self.mpm_model.grid_dim_x = self.mpm_model.n_grid
        self.mpm_model.grid_dim_y = self.mpm_model.n_grid
        self.mpm_model.grid_dim_z = self.mpm_model.n_grid
        (
            self.mpm_model.dx,
            self.mpm_model.inv_dx,
        ) = self.mpm_model.grid_lim / self.mpm_model.n_grid, float(
            self.mpm_model.n_grid / self.mpm_model.grid_lim
        )
        self.mpm_state.grid_m = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=float,
            device=device,
        )
        self.mpm_state.grid_v_in = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )
        self.mpm_state.grid_v_out = wp.zeros(
            shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid),
            dtype=wp.vec3,
            device=device,
        )

        if "E" in kwargs:
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[self.mpm_model.E, kwargs["E"]],
                device=device,
            )
        if "nu" in kwargs:
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[self.mpm_model.nu, kwargs["nu"]],
                device=device,
            )
        if "bulk_modulus" in kwargs:
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[self.mpm_model.bulk, kwargs["bulk_modulus"]],
                device=device
            )
        if "yield_stress" in kwargs:
            val = kwargs["yield_stress"]
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[self.mpm_model.yield_stress, val],
                device=device,
            )
        if "hardening" in kwargs:
            self.mpm_model.hardening = kwargs["hardening"]
        if "xi" in kwargs:
            self.mpm_model.xi = kwargs["xi"]
        if "friction_angle" in kwargs:
            self.mpm_model.friction_angle = kwargs["friction_angle"]
            sin_phi = wp.sin(self.mpm_model.friction_angle / 180.0 * 3.14159265)
            self.mpm_model.alpha = wp.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)

        if "g" in kwargs:
            self.mpm_model.gravitational_accelaration = wp.vec3(kwargs["g"][0], kwargs["g"][1], kwargs["g"][2])

        if "density" in kwargs:
            density_value = kwargs["density"]
            wp.launch(
                kernel=set_value_to_float_array,
                dim=self.n_particles,
                inputs=[self.mpm_state.particle_density, density_value],
                device=device,
            )
            wp.launch(
                kernel=get_float_array_product,
                dim=self.n_particles,
                inputs=[
                    self.mpm_state.particle_density,
                    self.mpm_state.particle_vol,
                    self.mpm_state.particle_mass,
                ],
                device=device,
            )
        if "rpic_damping" in kwargs:
            self.mpm_model.rpic_damping = kwargs["rpic_damping"]
        if "plastic_viscosity" in kwargs:
            self.mpm_model.plastic_viscosity = kwargs["plastic_viscosity"]
        if "softening" in kwargs:
            self.mpm_model.softening = kwargs["softening"]
        if "grid_v_damping_scale" in kwargs:
            self.mpm_model.grid_v_damping_scale = kwargs["grid_v_damping_scale"]

        if "additional_material_params" in kwargs:
            for params in kwargs["additional_material_params"]:
                param_modifier = MaterialParamsModifier()
                param_modifier.point = wp.vec3(params["point"])
                param_modifier.size = wp.vec3(params["size"])
                param_modifier.density = params["density"]
                param_modifier.E = params["E"]
                param_modifier.nu = params["nu"]
                wp.launch(
                    kernel=apply_additional_params,
                    dim=self.n_particles,
                    inputs=[self.mpm_state, self.mpm_model, param_modifier],
                    device=device,
                )

            wp.launch(
                kernel=get_float_array_product,
                dim=self.n_particles,
                inputs=[
                    self.mpm_state.particle_density,
                    self.mpm_state.particle_vol,
                    self.mpm_state.particle_mass,
                ],
                device=device,
            )


    def finalize_mu_lam_bulk(self, device = "cuda:0"):
        wp.launch(
            kernel=compute_mu_lam_from_E_nu, 
            dim = self.n_particles, 
            inputs = [self.mpm_state, self.mpm_model], 
            device=device
        )
        # wp.launch(
        #     kernel=compute_bulk,
        #     dim=self.n_particles,
        #     inputs=[self.mpm_state, self.mpm_model],
        #     device=device
        # )
        # print(self.mpm_model.lam)
        
    def p2g2p(self, step, dt, device="cuda:0"):
        grid_size = (
            self.mpm_model.grid_dim_x,
            self.mpm_model.grid_dim_y,
            self.mpm_model.grid_dim_z,
        )
        wp.launch(
            kernel=zero_grid,
            dim=(grid_size),
            inputs=[self.mpm_state, self.mpm_model],
            device=device,
        )
        wp.launch(
            kernel=zero_grid_instance,
            dim=((self.num_instances, self.mpm_model.grid_dim_x, self.mpm_model.grid_dim_y, self.mpm_model.grid_dim_z)),
            inputs=[self.mpm_state],
            device=device
        )
        # apply pre-p2g operations on particles
        for k in range(len(self.pre_p2g_operations)):
            wp.launch(
                kernel=self.pre_p2g_operations[k],
                dim=self.n_particles,
                inputs=[self.time, dt, self.mpm_state, self.impulse_params[k]],
                device=device,
            )
        # apply dirichlet particle v modifier
        for k in range(len(self.particle_velocity_modifiers)):
            wp.launch(
                kernel = self.particle_velocity_modifiers[k],
                dim = self.n_particles,
                inputs=[self.time, self.mpm_state, self.particle_velocity_modifier_params[k]],
                device=device,
            )

        # compute stress = stress(returnMap(F_trial))
        with wp.ScopedTimer(
            "compute_stress_from_F_trial",
            synchronize=True,
            print=False,
            dict=self.time_profile,
        ):
            wp.launch(
                kernel=compute_stress_from_F_trial,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model, dt],
                device=device,
            )  # F and stress are updated

        # p2g
        with wp.ScopedTimer(
            "p2g",
            synchronize=True,
            print=False,
            dict=self.time_profile,
        ):
            wp.launch(
                kernel=p2g_apic_with_stress,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model, dt],
                device=device,
            )  # apply p2g'

        # grid update
        with wp.ScopedTimer(
            "grid_update", synchronize=True, print=False, dict=self.time_profile
        ):
            wp.launch(
                kernel=grid_normalization_and_gravity,
                dim=(grid_size),
                inputs=[self.mpm_state, self.mpm_model, dt],
                device=device,
            )

        if self.mpm_model.grid_v_damping_scale < 1.0:
            wp.launch(
                kernel=add_damping_via_grid,
                dim=(grid_size),
                inputs=[self.mpm_state, self.mpm_model.grid_v_damping_scale],
                device=device,
            )

        # apply BC on grid
        # boundary conditions
        with wp.ScopedTimer(
            "apply_BC_on_grid", synchronize=True, print=False, dict=self.time_profile
        ):
            for k in range(len(self.grid_postprocess)):
                wp.launch(
                    kernel=self.grid_postprocess[k],
                    dim=grid_size,
                    inputs=[
                        self.time,
                        dt,
                        self.mpm_state,
                        self.mpm_model,
                        self.collider_params[k],
                    ],
                    device=device,
                )
                if self.modify_bc[k] is not None:
                    self.modify_bc[k](self.time, dt, self.collider_params[k])

        # g2p
        with wp.ScopedTimer(
            "g2p", synchronize=True, print=False, dict=self.time_profile
        ):
            wp.launch(
                kernel=g2p,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model, dt],
                device=device,
            )  # x, v, C, F_trial are updated

        #### CFL check ####
        particle_v = self.mpm_state.particle_v.numpy()
        if np.max(np.abs(particle_v)) > self.mpm_model.dx / dt:
            print("max particle v: ", np.max(np.abs(particle_v)))
            print("max allowed  v: ", self.mpm_model.dx / dt)
            print("does not allow v*dt>dx")
            input()
        #### CFL check ####

        # check valid
        self.time = self.time + dt
        wp.launch(
            kernel=get_J,
            dim=1,
            inputs=[self.mpm_state],
            outputs=[c_res],
            device=device,
        )
        # print(J)
        return c_res

    # set particle densities to all_particle_densities, 
    def reset_densities_and_update_masses(self, all_particle_densities, device = "cuda:0"):
        all_particle_densities = all_particle_densities.clone().detach()
        self.mpm_state.particle_density = torch2warp_float(all_particle_densities, dvc=device)
        wp.launch(
                kernel=get_float_array_product,
                dim=self.n_particles,
                inputs=[
                    self.mpm_state.particle_density,
                    self.mpm_state.particle_vol,
                    self.mpm_state.particle_mass,
                ],
                device=device,
            )

    # clone = True makes a copy, not necessarily needed
    def import_particle_x_from_torch(self, tensor_x, clone=True, device="cuda:0"):
        if tensor_x is not None:
            if clone:
                tensor_x = tensor_x.clone().detach()
            self.mpm_state.particle_x = torch2warp_vec3(tensor_x, dvc=device)

    # clone = True makes a copy, not necessarily needed
    def import_particle_v_from_torch(self, tensor_v, clone=True, device="cuda:0"):
        if tensor_v is not None:
            if clone:
                tensor_v = tensor_v.clone().detach()
            self.mpm_state.particle_v = torch2warp_vec3(tensor_v, dvc=device)

    # clone = True makes a copy, not necessarily needed
    def import_particle_F_from_torch(self, tensor_F, clone=True, device="cuda:0"):
        if tensor_F is not None:
            if clone:
                tensor_F = tensor_F.clone().detach()
            tensor_F = torch.reshape(tensor_F, (-1, 3, 3))  # arranged by rowmajor
            self.mpm_state.particle_F = torch2warp_mat33(tensor_F, dvc=device)

    # clone = True makes a copy, not necessarily needed
    def import_particle_C_from_torch(self, tensor_C, clone=True, device="cuda:0"):
        if tensor_C is not None:
            if clone:
                tensor_C = tensor_C.clone().detach()
            tensor_C = torch.reshape(tensor_C, (-1, 3, 3))  # arranged by rowmajor
            self.mpm_state.particle_C = torch2warp_mat33(tensor_C, dvc=device)

    def export_particle_x_to_torch(self):
        return wp.to_torch(self.mpm_state.particle_x)

    def export_particle_v_to_torch(self):
        return wp.to_torch(self.mpm_state.particle_v)

    def export_particle_F_to_torch(self):
        F_tensor = wp.to_torch(self.mpm_state.particle_F)
        F_tensor = F_tensor.reshape(-1, 9)
        return F_tensor

    def export_particle_R_to_torch(self, device="cuda:0"):
        with wp.ScopedTimer(
            "compute_R_from_F",
            synchronize=True,
            print=False,
            dict=self.time_profile,
        ):
            wp.launch(
                kernel=compute_R_from_F,
                dim=self.n_particles,
                inputs=[self.mpm_state, self.mpm_model],
                device=device,
            )

        R_tensor = wp.to_torch(self.mpm_state.particle_R)
        R_tensor = R_tensor.reshape(-1, 9)
        return R_tensor

    def export_particle_C_to_torch(self):
        C_tensor = wp.to_torch(self.mpm_state.particle_C)
        C_tensor = C_tensor.reshape(-1, 9)
        return C_tensor

    def export_particle_cov_to_torch(self, device="cuda:0"):
        if not self.mpm_model.update_cov_with_F:
            with wp.ScopedTimer(
                "compute_cov_from_F",
                synchronize=True,
                print=False,
                dict=self.time_profile,
            ):
                wp.launch(
                    kernel=compute_cov_from_F,
                    dim=self.n_particles,
                    inputs=[self.mpm_state, self.mpm_model],
                    device=device,
                )

        cov = wp.to_torch(self.mpm_state.particle_cov)
        return cov

    def print_time_profile(self):
        print("MPM Time profile:")
        for key, value in self.time_profile.items():
            print(key, sum(value))

    # given normal direction, say [0,0,1]
    # gradually release grid velocities from start position to end position
    def release_particles_sequentially(self, normal, start_position, end_position, num_layers, start_time, end_time):
        num_layers = 50
        point = [0,0,0]
        size = [0,0,0]
        axis = -1
        for i in range(3):
            if normal[i] == 0:
                point[i] = 1
                size[i] = 1
            else:
                axis = i
                point[i] = end_position
            
        half_length_portion = wp.abs(start_position - end_position)/num_layers
        end_time_portion = end_time / num_layers
        for i in range(num_layers):
            size[axis] = half_length_portion * (num_layers - i)
            self.enforce_particle_velocity_translation(point=point, size=size, velocity = [0,0,0], start_time=start_time, end_time=end_time_portion * (i+1))
