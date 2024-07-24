import numpy as np
import math
import random
from matplotlib import pyplot as plt
from IPython.display import clear_output

N = 10 # number of sphere
r = 0.4 + 0.6 * np.random.rand(N) # radii

# Upper/lower bound for guess coordinates
lower_bound = -5 + max(r)
upper_bound = 5 - max(r)

def get_rand_num(min_val, max_val):
    range = max_val - min_val
    choice = random.uniform(0,1)
    return min_val + range * choice

def eval(x, y, z):  # Evaluate maximum surface covered by sphere positions generated
    new_x, new_y, new_z = x.copy(), y.copy(), z.copy()  # Need new x and y and z in case we have overlap
    # Add radii dist to x and y pos
    edge_x = x.copy()
    edge_y = y.copy()
    edge_z = z.copy()
    ol = overlap(new_x, new_y, new_z)  # Check if we have overlap; returns [overlap: bool, index1: int/None, index2: int/None] where indices are overlaping spheres
    while ol[0]:
        # fig, ax = plt.subplots()
        # for i in range(N):
        #     if i == ol[1]:
        #         circ = plt.Circle((new_x[i], new_y[i]), r[i], color='r', fill=False)
        #     elif i == ol[2]:
        #         circ = plt.Circle((new_x[i], new_y[i]), r[i], color='g', fill=False)
        #     else:
        #         circ = plt.Circle((new_x[i], new_y[i]), r[i], color='b', fill=False)
        #     ax.add_patch(circ)
        # ax.set_xlim(-5 - max(r), 5 + max(r))
        # ax.set_ylim(-5 - max(r), 5 + max(r))
        # ax.set_aspect('equal')
        # plt.show()
        new_x[ol[1]] = get_rand_num(lower_bound, upper_bound)
        #new_x[ol[2]] = get_rand_num(lower_bound, upper_bound)
        new_y[ol[1]] = get_rand_num(lower_bound, upper_bound)
        #new_y[ol[2]] = get_rand_num(lower_bound, upper_bound)
        new_z[ol[1]] = get_rand_num(lower_bound, upper_bound)
        #new_z[ol[2]] = get_rand_num(lower_bound, upper_bound)

        # fig, ax = plt.subplots()
        # for i in range(N):
        #     if i == ol[1]:
        #         circ = plt.Circle((new_x[i], new_y[i]), r[i], color='r', fill=False)
        #     elif i == ol[2]:
        #         circ = plt.Circle((new_x[i], new_y[i]), r[i], color='g', fill=False)
        #     else:
        #         circ = plt.Circle((new_x[i], new_y[i]), r[i], color='b', fill=False)
        #     ax.add_patch(circ)
        # ax.set_xlim(-5 - max(r), 5 + max(r))
        # ax.set_ylim(-5 - max(r), 5 + max(r))
        # ax.set_aspect('equal')
        # plt.show()

        ol = overlap(new_x, new_y, new_z)


    for i in range(N):
        edge_x[i] = r[i] + new_x[i]
        edge_y[i] = r[i] + new_y[i]
        edge_z[i] = r[i] + new_z[i]

    return [new_x, new_y, new_z, (max(edge_x) - min(edge_x)) * (max(edge_y) - min(edge_y)) * (max(edge_z) - min(edge_z))]

# Returns [overlap: bool, index1: int/None, index2: int/None] where indices are overlaping spheres
def overlap(x, y, z):  # Implement no-overlap constraint
    for x1, y1, z1 in zip(x, y, z):
        i1 = x.index(x1)
        r1 = r[i1]
        for x2, y2, z2 in zip(x, y, z):
            if x1 == x2 or y1 == y2 or z1 == z2:  # Skip same coords
                continue

            p1 = np.array([x1, y1, z1])
            p2 = np.array([x2, y2, z2])

            squared_dist = np.sum((p1-p2)**2, axis=0)
            dist = np.sqrt(squared_dist)  # Distance between sphere1 and sphere2
            i2 = x.index(x2)
            r2 = r[i2]
            if dist < (r1 + r2):
                return [True, i1, i2]  # Dist < combined radii so must be overlapping, return which circles are overlapping

    return [False, None, None]  # No overlap


def crude_monte_carlo(num_samples=5000):
    min = np.inf
    min_x_set = []
    min_y_set = []
    min_z_set = []
    for i in range(num_samples):
        x = [get_rand_num(lower_bound, upper_bound) for i in range(N)]  # Generate set of x coords
        y = [get_rand_num(lower_bound, upper_bound) for i in range(N)]  # Generate set of y coords
        z = [get_rand_num(lower_bound, upper_bound) for i in range(N)]  # Generate set of z coords

        ex, ey, ez, e = eval(x, y, z)

        if min > e:
            print(i, min)
            min = e
            min_x_set = ex
            min_y_set = ey
            min_z_set = ez
        
        if i % 10000 == 0:
            print(i)

    return min, min_x_set, min_y_set, min_z_set

min, x, y, z = crude_monte_carlo(100000)
print(min)

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