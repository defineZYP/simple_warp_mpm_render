import torch
import imageio

import numpy as np
import warp as wp

from .postprocessor import adjust

@wp.struct
class HDRImage:
    img: wp.array(dtype=wp.vec3)
    width: int
    height: int

@wp.kernel
def hdr_process(
    hdr: HDRImage,
    exposure: float,
    gamma: float
):
    tid = wp.tid()
    hdr.img[tid] = adjust(
        hdr.img[tid],
        exposure,
        gamma
    )

@wp.func
def hdr_texture(
    hdr: HDRImage,
    uv: wp.vec2
):
    y = int(uv[1] * float(hdr.height))
    x = int(uv[0] * float(hdr.width))
    y = wp.clamp(y, 0, hdr.height - 1)
    x = wp.clamp(x, 0, hdr.width - 1)
    return hdr.img[y * hdr.width + x]


def init_hdr_image(hdr_path, device='cuda:0'):
    img = torch.tensor(imageio.v3.imread(hdr_path).astype(np.float32) / 255.0, device=device).squeeze(0)
    height = img.shape[0]
    width = img.shape[1]
    img = img.reshape(height * width, 3)
    hdr_img = wp.types.array(
        ptr=img.data_ptr(),
        dtype=wp.vec3,
        shape=img.shape[0],
        copy=False,
        # owner=False,
        requires_grad=img.requires_grad,
        # device=t.device.type)
        device=device,
    )
    hdr_img.tensor = img
    hdr = HDRImage()
    hdr.img = hdr_img
    hdr.width = width
    hdr.height = height
    return hdr

if __name__ == "__main__":
    hdr = init_hdr_image(f'/DATA/DATANAS1/zhangyip/phy/warp-mpm/assets/HDRi/0.hdr')
    print(hdr)
    wp.launch(
        kernel=hdr_process,
        dim=hdr.width * hdr.height,
        inputs=[hdr, 0.1, 0.1],
        device='cuda:0'
    )
    hdr_img = hdr.img.numpy()
    print(hdr_img.min(), hdr_img.max(), hdr_img.mean())