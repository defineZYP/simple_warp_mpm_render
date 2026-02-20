import os

import numpy as np
import cv2

def show_field_map(flow_path, save_path, clip_flow=None):
    flows = np.load(flow_path)['arr_0']

    for i in range(flows.shape[0]):
        flow = flows[i]
        h, w = flow.shape[:2]
        u = flow[..., 0]
        v = flow[..., 1]

        mag = np.sqrt(u**2 + v**2)
        ang = np.arctan2(v, u)

        # HSV image
        hsv = np.zeros((h, w, 3), dtype=np.uint8)

        # Hue: angle → [0,180]
        hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 180).astype(np.uint8)

        # Saturation
        hsv[..., 1] = 255

        # Value: magnitude normalization
        if clip_flow is not None:
            mag = np.clip(mag, 0, clip_flow)

        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = mag_norm.astype(np.uint8)

        # HSV → RGB
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite(os.path.join(save_path, f'flow_{i}.png'), rgb)

if __name__ == "__main__":
    # 使用
    show_field_map(
        './sim_results/0/force.npz',
        './tmp'
    )