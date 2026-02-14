import random

def scale_sample(low, high, scale):
    return random.randint(
        int(low * scale),
        int(high * scale)
    ) / scale

def get_random_material_from_range(material_range):
    material = {}
    scale = 10000
    for key in material_range:
        if isinstance(material_range[key], tuple) and len(material_range[key]) == 2:
            material[key] = scale_sample(material_range[key][0], material_range[key][1], scale)
        elif (
            isinstance(material_range[key], list) and 
            len(material_range[key]) >= 1 and 
            isinstance(material_range[key][0], tuple) and 
            len(material_range[key][0]) == 2
        ):
            sample = [scale_sample(material_range[key][i][0], material_range[key][i][1], scale) for i in range(len(material_range[key]))]
            material[key] = sample
        else:
            material[key] = material_range[key]
    return material

fluid_material_params_range = {
    'material': 'fluid',
    'bulk_modulus': 2.2e3,              # 越小越像泥浆，越大越像水，真实的水的体积模量约2.2e9，为了数值稳定性，减小为1/1000
    'friction_angle': 0.0,              # 越小越像水，越大越像泥浆
    'g': [0.0, -9.8, 0.0],              # 默认只受重力，这里没给9.8是为了数值的稳定性
    'density': 1000.0,                  # 水的密度固定为1000，基准密度结果
    # 'particle_dense': 1000000.0,        # 粒子密度
    'albedo': [175/255, 237/255, 250/255],
    'emission': [0.0, 0.0, 0.0],
    'roughness': 0.00,
    'metallic': 0.0,
    'transmission': 1.0,
    'ior': 1.333
}

jelly_material_params_range = {
    'material': 'jelly',
    'E': (1e3, 2e3),                                    # 杨氏模量，越大越硬
    'nu': (0.3, 0.35),                                  # 泊松比，越小越可压缩，越大越不可压缩
    'xi': (0.02, 0.2),                                  # 粘性
    'density': (1400, 1500),                            # 质量密度，果冻和水差不多
    # 'particle_dense': 1000000.0,                        # 粒子密度
    'g': [0.0, -9.8, 0.0],                                # 默认只受重力，这里没给9.8是为了数值的稳定性
    'albedo': [(0.3, 0.7), (0.3, 0.7), (0.3, 0.7)],
    'emission': [0.0, 0.0, 0.0],
    'roughness': (0.10, 0.25),
    'metallic': 0.0,
    'transmission': (0.75, 0.95),
    'ior': (1.45, 1.52)
}

mud_material_params_range = {                              # looks similar to jelly
    'material': 'jelly',
    'E': (7.5e2, 1e3),                                     # 杨氏模量，越大越硬
    'nu': (0.2, 0.25),                                     # 泊松比，越小越可压缩，越大越不可压缩
    'density': (5000, 7500),
    'rpic_damping': (0.15, 0.2),                           # 抑制震荡
    # 'particle_dense': 1000000.0,                           # 粒子密度
    'g': [0.0, -9.8, 0.0],                                   # 默认只受重力，这里没给9.8是为了数值的稳定性
    'albedo': [(0.35, 0.55), (0.25, 0.45), (0.15, 0.30)],
    'emission': [0.0, 0.0, 0.0],
    'roughness': (0.70, 0.90),
    'metallic': 0.0,
    'transmission': (0.0, 0.1),
    'ior': (1.30, 1.45)
}

rubber_ball_material_params_range = {
    'material': 'jelly',
    'E': (1e5, 5e5),                                    # 杨氏模量，越大越硬
    'nu': (0.4, 0.45),                                  # 泊松比，越小越可压缩，越大越不可压缩
    'density': (800, 1200),
    'rpic_damping': (0.05, 0.1),                        # 抑制震荡
    # 'particle_dense': 1000000.0,                        # 粒子密度
    'g': [0.0, -9.8, 0.0],                                # 默认只受重力，这里没给9.8是为了数值的稳定性
    'albedo': [(0.3, 0.7), (0.3, 0.7), (0.3, 0.7)],
    'emission': [0.0, 0.0, 0.0],
    'roughness': (0.35, 0.55),
    'metallic': 0.0,
    'transmission': (0.0, 0.02),
    'ior': (1.45, 1.52)
}

slime_material_params_range = {                         # looks similar to jelly
    'material': 'jelly',
    'E': (1e3, 2e3),                                  # 杨氏模量，越大越硬
    'nu': (0.2, 0.25),                                  # 泊松比，越小越可压缩，越大越不可压缩
    'density': (1000, 1500),
    'rpic_damping': (0.05, 0.1),                        # 抑制震荡
    # 'particle_dense': 1000000.0,                        # 粒子密度
    'g': [0.0, -9.8, 0.0],                                # 默认只受重力，这里没给9.8是为了数值的稳定性
    'albedo': [(0.3, 0.7), (0.3, 0.7), (0.3, 0.7)],
    'emission': [0.0, 0.0, 0.0],
    'roughness': (0.05, 0.25),
    'metallic': 0.0,
    'transmission': (0.50, 0.85),
    'ior': (1.40, 1.48)
}


foam_material_params_range = {                          # looks similar to jelly
    'material': 'foam',
    'E': (2e2, 4e2),                                    # 杨氏模量，越大越硬
    'nu': (0.2, 0.25),                                  # 泊松比，越小越可压缩，越大越不可压缩
    'density': (20, 50),
    'yield_stress': (5e4, 1e5),                         # 屈服应力，塑性变形
    'rpic_damping': 0.25,                               # 抑制震荡
    # 'particle_dense': 1000000.0,                        # 粒子密度
    'plastic_viscosity': (1, 10),                       # 塑形粘度
    'g': [0.0, -9.8, 0.0],                                # 默认只受重力，这里没给9.8是为了数值的稳定性
    'albedo': [(0.7, 0.9), (0.7, 0.9), (0.3, 0.5)],
    'emission': [0.0, 0.0, 0.0],
    'roughness': (0.8, 0.95),
    'metallic': 0.0,
    'transmission': (0.0, 0.02),
    'ior': (1.3, 1.45)
}

sand_material_params_range = {                          # looks similar to jelly
    'material': 'sand',
    'E': (5e4, 2e5),                                  # 杨氏模量，越大越硬
    'nu': (0.2, 0.225),                                 # 泊松比，越小越可压缩，越大越不可压缩
    'friction_angle': 40.0,
    'density': (1300.0, 1800.0),
    # 'particle_dense': 2000000.0,                         # 粒子密度
    'rpic_damping': 0.05,
    'g': [0.0, -9.8, 0.0],                                # 默认只受重力，这里没给9.8是为了数值的稳定性
    'albedo': [(0.8, 0.95), (0.7, 0.85), (0.4, 0.5)],
    'emission': [0.0, 0.0, 0.0],
    'roughness': (0.85, 0.95),
    'metallic': 0.0,
    'transmission': 0.0,
    'ior': 1.55
}

gold_material_params_range = {
    'material': 'metal',
    'E': (7.5e6, 8e6),
    'nu': (0.4, 0.45),
    'yield_stress': (1e8, 3e8),                         # 黄金美妙的延展性（
    'xi': 0.05,
    'hardening': 0,
    # 'particle_dense': 2000000.0,
    'rpic_damping': 0.0,
    'density': 19300.0,
    'g': [0.0, -9.8, 0.0],                                # 默认只受重力，这里没给9.8是为了数值的稳定性
    'albedo': [(0.95, 1.00), (0.8, 0.9), (0.0, 0.05)],
    'emission': [0.0, 0.0, 0.0],
    'roughness': 0.0,
    'metallic': (0.95, 1.0),
    'transmission': 0.0,
    'ior': (0.15, 0.35)                                 # bushiyong
}

# gold_material_params_range = {
#     'material': 'metal', 
#     'E': 7521505.05, 
#     'nu': 0.4411, 
#     'yield_stress': 182184477.1068, 
#     'xi': 0.05, 
#     'hardening': 0, 
#     'particle_dense': 2000000.0, 
#     'rpic_damping': 0.5, 
#     'density': 19300.0, 
#     'g': [0.0, -9.8, 0.0], 
#     'albedo': [0.961, 0.8325, 0.0158], 
#     'emission': [0.0, 0.0, 0.0], 
#     'roughness': 0.0, 
#     'metallic': 0.1, 
#     'transmission': 0.0, 
#     'ior': 0.47
# }

silver_material_params_range = {
    'material': 'metal',
    'E': (8.1e6, 8.5e6),
    'nu': (0.35, 0.4),
    'yield_stress': (5e7, 6e7),                         # 黄金美妙的延展性（
    'xi': 0.01,
    'hardening': 1.0,
    # 'particle_dense': 2000000.0,
    'rpic_damping': 0.0,
    'density': 10490.0,
    'g': [0.0, -9.8, 0.0],                                # 默认只受重力，这里没给9.8是为了数值的稳定性
    'albedo': [(0.85, 0.90), (0.85, 0.9), (0.85, 0.9)],
    'emission': [0.0, 0.0, 0.0],
    'roughness': (0.05, 0.35),
    'metallic': (0.95, 1.0),
    'transmission': 0.0,
    'ior': (0.15, 0.35)                                 # bushiyong
}

# iron_material_params_range = {'material': 'metal', 'E': 20630840.721, 'nu': 0.3237, 'yield_stress': 256841489.8631, 'xi': 0.01, 'hardening': 1.0, 'particle_dense': 2000000.0, 'rpic_damping': 0.2, 'density': 7870.0, 'g': [0.0, -9.8, 0.0], 'albedo': [0.4925, 0.4782, 0.4601], 'emission': [0.0, 0.0, 0.0], 'roughness': 0.4983, 'metallic': 0.8972, 'transmission': 0.0, 'ior': 0.2496}

iron_material_params_range = {
    'material': 'metal',
    'E': (2e8, 2.1e8),
    'nu': (0.29, 0.34),
    'yield_stress': (2.5e8, 2.6e8),                         # 黄金美妙的延展性（
    'xi': 0.0,
    'hardening': 1.0,
    # 'particle_dense': 2000000.0,
    'rpic_damping': 0.0,
    'density': 7870.0,
    'g': [0.0, -9.8, 0.0],                                # 默认只受重力，这里没给9.8是为了数值的稳定性
    'albedo': [(0.45, 0.50), (0.45, 0.5), (0.45, 0.5)],
    'emission': [0.0, 0.0, 0.0],
    'roughness': (0.25, 0.65),
    'metallic': (0.85, 1.0),
    'transmission': 0.0,
    'ior': (0.15, 0.35)                                 # bushiyong
}

copper_material_params_range = {
    'material': 'metal',
    'E': (1e8, 1.1e8),
    'nu': (0.34, 0.38),
    'yield_stress': (7.0e7, 7.1e7),                         # 黄金美妙的延展性（
    'xi': 0.01,
    'hardening': 1.0,
    # 'particle_dense': 2000000.0,
    'rpic_damping': 0.0,
    'density': 8960.0,
    'g': [0.0, -9.8, 0.0],                                # 默认只受重力，这里没给9.8是为了数值的稳定性
    'albedo': [(0.75, 0.80), (0.45, 0.5), (0.2, 0.25)],
    'emission': [0.0, 0.0, 0.0],
    'roughness': (0.1, 0.55),
    'metallic': (0.95, 1.0),
    'transmission': 0.0,
    'ior': (0.15, 0.35)                                 # bushiyong
}

metal_material_params_range = {
    'material': 'metal',
    'E': (1e6, 1e9),
    'mu': (0.3, 0.35),
    'yield_stress': (2e4, 1e6),
    'xi': 5e4,
    'hardening': 0,
    # 'particle_dense': 2000000.0,
    'rpic_damping': 0.5,
    'density': (7000, 15000),
    'g': [0.0, -9.8, 0.0],                                # 默认只受重力，这里没给9.8是为了数值的稳定性
    'albedo': [(0.3, 0.7), (0.3, 0.7), (0.3, 0.7)],
    'emission': [0.0, 0.0, 0.0],
    'roughness': (0.35, 0.55),
    'metallic': (0.85, 1.0),
    'transmission': 0.0,
    'ior': (1.45, 1.52)
}

plasticine_material_params_range = {
    'material': 'plasticine',
    'E': (1e6, 1e7),
    'mu': (0.25, 0.35),
    'yield_stress': (3.9e7, 4.1e7),
    'xi': 0.1,
    'hardening': 0,
    'softening': 0.1,
    'density': (1200, 2000),
    # 'particle_dense': 2000000.0,
    'rpic_damping': 0.5,
    'g': [0.0, -4.0, 0.0],
    'albedo': [(0.3, 0.7), (0.3, 0.7), (0.3, 0.7)],
    'emission': [0.0, 0.0, 0.0],
    'roughness': (0.3, 0.85),
    'metallic': 0.0,
    'transmission': (0.0, 0.05),
    'ior': (1.45, 1.55)
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
    jelly_material_params_range,
    mud_material_params_range,
    rubber_ball_material_params_range,
    slime_material_params_range,
    foam_material_params_range,
    sand_material_params_range,
    gold_material_params_range,
    silver_material_params_range,
    iron_material_params_range,
    copper_material_params_range,
    metal_material_params_range,
    plasticine_material_params_range,
    # ground_material_params_range
]

materials_mapping = {
    'fluid': 0,
    'jelly': 1,
    'mud': 2,
    'rubber_ball': 3,
    'slime': 4,
    'foam': 5,
    'sand': 6,
    'gold': 7,
    'silver': 8,
    'iron': 9,
    'copper': 10,
    'metal': 11,
    'plasticine': 12,
    # 'ground': 13
}

materials_trans_mapping = {}
for k in materials_mapping:
    materials_trans_mapping[materials_mapping[k]] = k