import json
from scene_init.materials import materials_range

if __name__ == "__main__":
    i = 0
    for i in range(2500):
        with open(f'/DATA/DATANAS2/zhangyip/sim_results/{i}/scene.json', 'r') as f:
            true_scene = json.load(f)
            for idx in range(len(true_scene)):
                true_material = true_scene[idx]['material']
                match = False
                # print(true_material)
                for material_range in materials_range:
                    find = True
                    for key in material_range:
                        if key in ["name", "g", "particle_dense", "E", "nu", "yield_stress", "xi", "rpic_damping", "bulk_modulus"]:
                            continue
                        if key not in true_material:
                            print(key)
                            find = False
                            break
                        true_value = true_material[key]
                        value_range = material_range[key]
                        if isinstance(true_value, str):
                            if true_value != value_range:
                                print(key, true_value, value_range)
                                find = False
                                break
                        if isinstance(true_value, float):
                            if isinstance(value_range, float):
                                if true_value != value_range:
                                    print(key, true_value, value_range)
                                    find = False
                                    break
                            elif isinstance(value_range, tuple):
                                if true_value < value_range[0] or true_value > value_range[1]:
                                    print(key, true_value, value_range)
                                    find = False
                                    break
                            else:
                                find = False
                                break
                        if isinstance(true_value, list):
                            if isinstance(value_range, list) and isinstance(value_range[0], float):
                                for v, t in zip(value_range, true_value):
                                    if v != t:
                                        print(key, true_value, value_range)
                                        find = False
                                if not find:
                                    break
                            elif isinstance(value_range, list) and isinstance(value_range[0], tuple):
                                for v, t in zip(value_range, true_value):
                                    if t < v[0] or t > v[1]:
                                        print(key, true_value, value_range)
                                        find = False
                                if not find:
                                    break
                            else:
                                find = False
                                break
                    if find:
                        match = True
                        true_scene[idx]['material']['name'] = material_range['name']
                        break
                if not match:
                    raise ValueError(f"FIND PROBELM WITH {i}")
        with open(f'/DATA/DATANAS2/zhangyip/sim_results/{i}/scene_resolve.json', 'w') as f:
            json.dump(true_scene, f, indent=4)
