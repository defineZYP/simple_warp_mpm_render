import taichi as ti

import numpy as np
import torch

img = ti.tools.imread('./assets/HDRi/10.hdr')
print(type(img), img.shape)
print(img[2367, 2048 - 1102])

import imageio.v2 as imageio

img = imageio.imread('./assets/HDRi/10.hdr', format="HDR")
print(type(img), img.shape)
print(img[1102, 2367])

import cv2

img = cv2.imread('./assets/HDRi/10.hdr', cv2.IMREAD_UNCHANGED)

print(type(img), img.shape)   # float32
print(img[1102, 2367])
