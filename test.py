import numpy as np
import torch
import matplotlib.pyplot as plt

points = []
T = 10000
omega = 2 * np.pi / T
delta_t = 1 / T
theta = 0
v = np.array([0., 1.])
point = np.array([1., 0.])
# for t in range(1, T + 1):
#     points.append(point)
#     # update
#     point += delta_t * v
#     theta = 2 * np.pi * t / T
#     v = np.array((-np.sin(theta), np.cos(theta)))

import numpy as np

m = 1.0
r = 1.0
omega_max = 2*np.pi

steps = 20000
dt = 1/steps

T_acc = steps // 2
alpha = omega_max / (T_acc*dt)

x = np.array([r, 0.0])
v = np.array([0.0, 0.0])

points = []

for i in range(steps):

    radius = np.linalg.norm(x)
    r_hat = x / radius
    t_hat = np.array([-r_hat[1], r_hat[0]])

    if i < T_acc:
        omega = alpha * i * dt
        tangential = alpha * r
    else:
        omega = omega_max
        tangential = 0.0

    a = -omega**2 * r_hat * r + tangential * t_hat
    F = m * a

    # symplectic update
    v += (F/m) * dt
    x += v * dt

    # optional radius correction
    x = r * x / np.linalg.norm(x)

    points.append(x.copy())

pts = np.vstack(points)  # shape: (N, 2)

plt.scatter(pts[:, 0], pts[:, 1])
plt.savefig('./output.png')