# -*- coding: utf-8 -*-
"""Untitled11.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1qkr1uZyWv1u3pzJSs0VQ9X63ZaW3jS0E
"""

import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return x * np.exp(-x**2 - y**2)

plotN = 200
x = np.linspace(-1, 1, plotN)
y = np.linspace(-1, 1, plotN)

x, y = np.meshgrid(x, y)
z = f(x, y)

fig = plt.figure(figsize=(10, 5))

# Primera subfigura
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(x, y, z, cmap='jet', shade=False)
ax1.set_title('Superficie 3D')

# Segunda subfigura
ax2 = fig.add_subplot(122)
contour = ax2.contour(x, y, z, levels=20, cmap='jet')
ax2.set_title('Contorno')
plt.colorbar(contour, ax=ax2)

plt.show()

# Algoritmo de optimización
N = 100
A = 1
r = 0.5
Qmin = 0
Qmax = 2
alpha = 0.9
gamma = 0.9
Lb = -2
Ub = 2

x = np.random.uniform(Lb, Ub, (N, 2))
v = np.zeros((N, 2))
f_x = np.apply_along_axis(lambda t: f(t[0], t[1]), 1, x)
f_min = np.inf
f_min_pos = np.zeros(2)

for i in range(N):
    if f_x[i] < f_min:
        f_min = f_x[i]
        f_min_pos = x[i]

iter = 0
while iter < 1000:
    Q = Qmin + (Qmax - Qmin) * np.random.rand(N)
    v = v + (x - f_min_pos) * np.tile(Q, (2, 1)).T
    x = x + v

    for i in range(N):
        if np.random.rand() > gamma:
            x[i] = f_min_pos + A * np.random.randn(2)
        if np.random.rand() < r and f(x[i, 0], x[i, 1]) < f_x[i]:
            f_x[i] = f(x[i, 0], x[i, 1])
            x[i] = x[i] + A * np.random.randn(2)

    for i in range(N):
        if f_x[i] < f_min:
            f_min = f_x[i]
            f_min_pos = x[i]

    Qmax = alpha * Qmax
    iter += 1

print("Mínimo valor de la función:", f_min)
print("Posición del mínimo:", f_min_pos)