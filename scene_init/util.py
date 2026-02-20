import random

def overlap3d(a, b):
    ax1, ay1, az1, ax2, ay2, az2 = a
    bx1, by1, bz1, bx2, by2, bz2 = b
    
    return not (
        ax2 <= bx1 or bx2 <= ax1 or
        ay2 <= by1 or by2 <= ay1 or
        az2 <= bz1 or bz2 <= az1
    )

def sample_collide_boxes_3d(bounding_box=(0.1, 0.9)):
    base = 1 / 256
    target_tower = random.randint(2, 4)
    # target_tower = 1
    source_tower = random.randint(2, target_tower)

    source_boxes = []
    base_height = bounding_box[0]

    for _ in range(source_tower):
        # w = random.uniform(bounding_box[0], 0.24)
        w = random.randint(10, 15) * 2 * base
        # h = random.uniform(bounding_box[0], 0.2)
        h = random.randint(5, 10) * 2 * base
        # d = random.uniform(bounding_box[0], 0.5)
        d = random.uniform(10, 15) * 2 * base

        cx = 60 * base
        cy = h / 2 + base_height
        cz = 0.5

        base_height += h + 0.01

        source_boxes.append([cx, cy, cz, w, h, d])

    target_boxes = []
    base_height = bounding_box[0]
    for _ in range(target_tower):
        # w = random.uniform(bounding_box[0], 0.24)
        w = random.randint(10, 20) * 2 * base
        # h = random.uniform(bounding_box[0], 0.2)
        h = random.randint(10, 20) * 2 * base
        # d = random.uniform(bounding_box[0], 0.5)
        d = random.uniform(10, 25) * 2 * base

        cx = 150 * base
        cy = h / 2 + base_height
        cz = 0.5

        base_height += h + 0.01

        target_boxes.append([cx, cy, cz, w, h, d])
    
    return source_boxes, target_boxes

def sample_boxes_3d(n,
                    size_range=(0.15, 0.4),
                    bounding_box=(0.1, 0.9),
                    max_trials=20000):
    
    boxes = []
    
    for _ in range(n):
        for _ in range(max_trials):
            w = random.uniform(*size_range)
            h = random.uniform(*size_range)
            d = random.uniform(*size_range)
            
            cx = random.uniform(w/2 + bounding_box[0], bounding_box[1] - w/2)
            cy = random.uniform(h/2 + bounding_box[0], bounding_box[1] - h/2)
            cz = random.uniform(d/2 + bounding_box[0], bounding_box[1] - d/2)
            
            rect = (
                cx - w/2, cy - h/2, cz - d/2,
                cx + w/2, cy + h/2, cz + d/2
            )
            
            if all(not overlap3d(rect, b) for b in boxes):
                boxes.append(rect)
                break
    
    # 转回 center-size
    result = []
    for x1,y1,z1,x2,y2,z2 in boxes:
        result.append((
            (x1+x2)/2,
            (y1+y2)/2,
            (z1+z2)/2,
            x2-x1,
            y2-y1,
            z2-z1
        ))
    
    return result

_mesh_mapping = {
    'apple': 0,
    'ball': 15,
    'bed': 14,
    'bench': 1,
    'bread': 2,
    'carrot': 3,
    'couch': 4,
    'dragon': 5,
    'futou': 6,
    'knife': 13 ,
    'plant': 12 ,
    'rose': 11,
    'tower': 7,
    'tree1': 10,
    'tree2': 9,
    'tree3': 8
}

mesh_mapping = {}
for k in _mesh_mapping:
    mesh_mapping[_mesh_mapping[k]] = k