import numpy as np
import math
import random
from matplotlib import pyplot as plt
from IPython.display import clear_output

N = 10 # number of sphere
r = 0.4 + 0.6 * np.random.rand(N) # radii
intens = np.random.rand(N, N)
np.fill_diagonal(intens, 0)

# Upper/lower bound for guess coordinates
lower_bound = -5 + max(r)
upper_bound = 5 - max(r)

def get_rand_num(min_val, max_val):
    range = max_val - min_val
    choice = random.uniform(0,1)
    return min_val + range * choice


 # Eval 3d distance between spheres at index i1 and i2
def sphere_dist(x, y, z, i1, i2):
    p1 = np.array([x[i1], y[i1], z[i1]])
    p2 = np.array([x[i2], y[i2], z[i2]])
    squared_dist = np.sum((p1-p2)**2, axis=0)
    dist = np.sqrt(squared_dist)  # Distance between sphere1 and sphere2

    return dist

def eval(x, y, z):  # Evaluate maximum surface covered by sphere positions generated
    new_x, new_y, new_z = x.copy(), y.copy(), z.copy()  # Need new x and y and z in case we have overlap

    ol = overlap(new_x, new_y, new_z)  # Check if we have overlap; returns [overlap: bool, index1: int/None, index2: int/None] where indices are overlaping spheres
    while ol[0]:
        new_x[ol[1]] = get_rand_num(lower_bound, upper_bound)
        new_y[ol[1]] = get_rand_num(lower_bound, upper_bound)
        new_z[ol[1]] = get_rand_num(lower_bound, upper_bound)

        ol = overlap(new_x, new_y, new_z)

    inv_dists = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            inv_dists[i,j] = 1/sphere_dist(x, y, z, i, j)

    efficiency = np.sum(np.dot(inv_dists, intens))

    return [new_x, new_y, new_z, efficiency]

# Returns [overlap: bool, index1: int/None, index2: int/None] where indices are overlaping spheres
def overlap(x, y, z):  # Implement no-overlap constraint
    for x1, y1, z1 in zip(x, y, z):
        i1 = x.index(x1)
        r1 = r[i1]
        for x2, y2, z2 in zip(x, y, z):
            if x1 == x2 or y1 == y2 or z1 == z2:  # Skip same coords
                continue

            i2 = x.index(x2)
            r2 = r[i2]

            dist = sphere_dist(x, y, z, i1, i2)
            if dist < (r1 + r2):
                return [True, i1, i2]  # Dist < combined radii so must be overlapping, return which circles are overlapping

    return [False, None, None]  # No overlap


def crude_monte_carlo(num_samples=5000):
    max_efficiency = 0
    max_x_set = []
    max_y_set = []
    max_z_set = []
    for i in range(num_samples):
        x = [get_rand_num(lower_bound, upper_bound) for i in range(N)]  # Generate set of x coords
        y = [get_rand_num(lower_bound, upper_bound) for i in range(N)]  # Generate set of y coords
        z = [get_rand_num(lower_bound, upper_bound) for i in range(N)]  # Generate set of z coords

        ex, ey, ez, e = eval(x, y, z)

        if max_efficiency < e:
            print(i, max_efficiency)
            max_efficiency = e
            max_x_set = ex
            max_y_set = ey
            max_z_set = ez
        
        if i % 10000 == 0:
            print(i)

    return max_efficiency, max_x_set, max_y_set, max_z_set

max_efficiency, x, y, z = crude_monte_carlo(100000)
print(max_efficiency)

fig = plt.figure(figsize=(5, 5))
ax = fig.gca(projection='3d')
ax.set_aspect('auto')
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
for i in range(N):
    ax.plot_surface(
        z[i] + r[i] * np.cos(u) * np.sin(v), y[i] + r[i] * np.sin(u) * np.sin(v), z[i] + r[i] * np.cos(v), color="b"
    )
ax.set_xlim(-5 - max(r), 5 + max(r))
ax.set_ylim(-5 - max(r), 5 + max(r))
ax.set_zlim(-5 - max(r), 5 + max(r))
plt.show()