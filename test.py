import warp as wp
import numpy as np

import torch

from render.scene import init_hdr_image, hdr_texture, hdr_process, HDRImage
from warp_utils import torch2warp_vec2

height = 2048
width = 4096

uvs = []

def sample_uv(v):
    v = v / np.linalg.norm(v)
    phi = np.atan2(v[2], v[0])
    theta = np.asin(np.clip(v[1], -1.0, 1.0))
    _u = 0.5 + phi * (0.5 / np.pi)
    _v = 0.5 - theta * (1.0 / np.pi)
    uv = np.array([_u, _v])
    return uv

num = 300

for i in range(num):
    for j in range(num):
        phi = 2 * np.pi / num * i
        theta = np.pi / num * j

        r = theta * np.sin(theta)
        y = theta * np.cos(theta)
        x = r * np.cos(phi)
        z = r * np.sin(phi)
        d = np.array([x, y, z])
        uvs.append(sample_uv(d))

uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')
uvs = torch2warp_vec2(uvs, 'cuda:0')

colors = wp.zeros(
    shape=10000,
    dtype=wp.vec3,
    device='cuda:0'
)

@wp.kernel
def test(
    hdr: HDRImage,
    uvs: wp.array(dtype=wp.vec2),
    colors: wp.array(dtype=wp.vec3)
):
    tid = wp.tid()
    uv = uvs[tid]
    color = hdr_texture(hdr, uv)
    colors[tid] = color

hdr = init_hdr_image(
    './assets/HDRi/0.hdr',
    'cuda:0'
)

wp.launch(
    kernel=hdr_process,
    dim=hdr.width * hdr.height,
    inputs=[hdr, 1.0, 2.2],
    device='cuda:0'
)

wp.launch(
    kernel=test,
    dim=num * num,
    inputs=[hdr, uvs],
    outputs=[colors],
    device='cuda:0'
)
colors = colors.numpy()
print(colors.min(), colors.max(), colors.mean())