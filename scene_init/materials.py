import random

def get_random_material_from_range(material_range):
    material = {}
    for key in material_range:
        if isinstance(material_range[key], tuple) and len(material_range[key]) == 2:
            material[key] = random.uniform(material_range[key][0], material_range[key][1])
        else:
            material[key] = material_range[key]
    return material

fluid_material_params_range = {
    'material': 'fluid',
    'bulk_modulus': 2.2e6,              # 越小越像泥浆，越大越像水，真实的水的体积模量约2.2e9，为了数值稳定性，减小为1/1000
    'friction_angle': 0.0,              # 越小越像水，越大越像泥浆
    'g': [0.0, -4.0, 0.0],              # 默认只受重力，这里没给9.8是为了数值的稳定性
    'density': 1000.0,                   # 水的密度固定为1000，基准密度结果
    'particle_dense': 1000000.0,        # 粒子密度
    'color': [175/255, 237/255, 250/255],
    'albedo': [175/255, 237/255, 250/255],
    'emission': [0.0, 0.0, 0.0],
    'roughness': 0.02,
    'metallic': 0.0,
    'transmission': 1.0,
    'ior': 1.333
}

# mud_material_params_range = {
#     'material': 'fluid',
#     'bulk_modulus': (100.0, 200.0),     # 越小越像泥浆，越大越像水
#     'friction_angle': (10.0, 30.0),     # 越小越像水，越大越像泥浆
#     'g': [0.0, -4.0, 0.0],              # 默认只受重力，这里没给9.8是为了数值的稳定性
#     'particle_dense': 1000000.0,        # 粒子密度
#     'density': (1500.0, 2000.0)         # 泥巴比水重
# }

jelly_material_params_range = {
    'material': 'jelly',
    'E': (1e3, 2e3),                                    # 杨氏模量，越大越硬
    'nu': (0.3, 0.35),                                  # 泊松比，越小越可压缩，越大越不可压缩
    'xi': (0.02, 0.2),                                  # 粘性
    'density': (1400, 1500),                            # 质量密度，果冻和水差不多
    'particle_dense': 1000000.0,                        # 粒子密度
    'g': [0.0, -4, 0.0],                                # 默认只受重力，这里没给9.8是为了数值的稳定性
    'color': [227/255, 197/255, 120/255],
    'albedo': [227/255, 197/255, 120/255],
    'emission': [0.0, 0.0, 0.0],
    'roughness': 0.15,
    'metallic': 0.0,
    'transmission': 0.0,
    'ior': 1.45
}

rubber_ball_material_params_range = {
    'material': 'jelly',
    'E': (1e5, 5e5),                                    # 杨氏模量，越大越硬
    'nu': (0.4, 0.45),                                  # 泊松比，越小越可压缩，越大越不可压缩
    'density': (800, 1200),
    'rpic_damping': (0.05, 0.1),                        # 抑制震荡
    'particle_dense': 1000000.0,                        # 粒子密度
    'g': [0.0, -4, 0.0],                                # 默认只受重力，这里没给9.8是为了数值的稳定性
}

slime_material_params_range = {                         # looks similar to jelly
    'material': 'jelly',
    'E': (1e3, 2e3),                                  # 杨氏模量，越大越硬
    'nu': (0.2, 0.25),                                  # 泊松比，越小越可压缩，越大越不可压缩
    'density': (1000, 1500),
    'rpic_damping': (0.05, 0.1),                        # 抑制震荡
    'particle_dense': 1000000.0,                        # 粒子密度
    'g': [0.0, -4, 0.0],                                # 默认只受重力，这里没给9.8是为了数值的稳定性
}

mud_material_params_range = {                           # looks similar to jelly
    'material': 'jelly',
    'E': (7.5e2, 1e3),                                  # 杨氏模量，越大越硬
    'nu': (0.2, 0.25),                                  # 泊松比，越小越可压缩，越大越不可压缩
    'density': (5000, 7500),
    'rpic_damping': (0.15, 0.2),                        # 抑制震荡
    'particle_dense': 1000000.0,                        # 粒子密度
    'g': [0.0, -4, 0.0],                                # 默认只受重力，这里没给9.8是为了数值的稳定性
}

foam_material_params_range = {                          # looks similar to jelly
    'material': 'foam',
    'E': (2e2, 4e2),                                    # 杨氏模量，越大越硬
    'nu': (0.2, 0.25),                                  # 泊松比，越小越可压缩，越大越不可压缩
    'density': (20, 50),
    'yield_stress': (5e4, 1e5),                         # 屈服应力，塑性变形
    'rpic_damping': 0.25,                               # 抑制震荡
    'particle_dense': 1000000.0,                        # 粒子密度
    'plastic_viscosity': (1, 10),                       # 塑形粘度
    'g': [0.0, -4, 0.0],                                # 默认只受重力，这里没给9.8是为了数值的稳定性
}

sand_material_params_range = {                          # looks similar to jelly
    'material': 'sand',
    'E': (1e6, 1e6+1),                                  # 杨氏模量，越大越硬
    'nu': (0.2, 0.225),                                 # 泊松比，越小越可压缩，越大越不可压缩
    'friction_angle': 40.0,
    'density': (1300.0, 1800.0),
    'particle_dense': 2000000.0,                        # 粒子密度
    'rpic_damping': 0.5,
    'g': [0.0, -4, 0.0],                                # 默认只受重力，这里没给9.8是为了数值的稳定性
}

metal_material_params_range = {
    'material': 'metal',
    'E': (1e6, 1e7),
    'mu': (0.3, 0.35),
    'yield_stress': (2e4, 1e6),
    'xi': 5e4,
    'hardening': 0,
    'particle_dense': 2000000.0,
    'rpic_damping': 0.5,
    'density': (7000, 15000),
    'g': [0.0, -4, 0.0],                                # 默认只受重力，这里没给9.8是为了数值的稳定性
}

plasticine_material_params_range = {
    'material': 'plasticine',
    'E': (1e5, 5e5),
    'mu': (0.25, 0.3),
    'yield_stress': (800, 2500),
    'xi': 1e3,
    'hardening': 0,
    'softening': -0.5,
    'density': (1200, 2000),
    'particle_dense': 2000000.0,
    'rpic_damping': 0.5,
    'g': [0.0, -4.0, 0.0]
}

ground_material_params_range = {
    'material': 'sand',
    'color': [0.72, 0.72, 0.72],
    'albedo': [0.72, 0.72, 0.72],
    'emission': [0.0, 0.0, 0.0],
    'roughness': 0.25,
    'metallic': 0.0,
    'transmission': 0.0,
    'ior': 1.52
}

materials_range = [
    fluid_material_params_range,
    # mud_material_params_range,
    jelly_material_params_range,
    # rubber_ball_material_params_range,
    # slime_material_params_range,
    # foam_material_params_range,
    # sand_material_params_range,
    # metal_material_params_range,
    # plasticine_material_params_range,
    ground_material_params_range
]

materials_mapping = {
    'fluid': 0,
    'mud': 1,
    'jelly': 1,
    'rubber_ball': 3,
    'slime': 4,
    'foam': 5,
    'sand': 6,
    'metal': 7,
    'plasticine': 8,
    'ground': -1
}

