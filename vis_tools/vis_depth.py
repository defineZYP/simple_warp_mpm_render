import cv2
import os
import numpy as np

def show_depth_map(np_path, save_path):
    depth = np.load(np_path)
    for i in range(depth.shape[0]):
        d = depth[i]
        d[d == 1e10] = 2
        dmin = np.min(d)
        dmax = np.max(d)
        depth_norm = (d - dmin) / (dmax - dmin + 1e-8)
        depth_img = (depth_norm * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(save_path, f'depth_gray_{i}.png'), depth_img)


if __name__ == "__main__":
    show_depth_map(
        '/DATA/DATANAS1/zhangyip/phy/warp-mpm/sim_results/ball/depth.npy',
        './tmp'
    )