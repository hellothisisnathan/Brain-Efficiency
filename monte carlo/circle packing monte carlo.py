import numpy as np
import math
import random
from matplotlib import pyplot as plt
from IPython.display import clear_output

N = 10 # number of circles
r = 0.4 + 0.6 * np.random.rand(N) # radii

# Upper/lower bound for guess coordinates
lower_bound = -5 + max(r)
upper_bound = 5 - max(r)

def get_rand_num(min_val, max_val):
    range = max_val - min_val
    choice = random.uniform(0,1)
    return min_val + range * choice

def eval(x, y):  # Evaluate maximum surface covered by sphere positions generated
    new_x, new_y = x.copy(), y.copy()  # Need new x and y in case we have overlap
    # Add radii dist to x and y pos
    edge_x = x.copy()
    edge_y = y.copy()
    ol = overlap(new_x, new_y)  # Check if we have overlap; returns [overlap: bool, index1: int/None, index2: int/None] where indices are overlaping spheres
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

        ol = overlap(new_x, new_y)


    for i in range(N):
        edge_x[i] = r[i] + new_x[i]
        edge_y[i] = r[i] + new_y[i]

    return [new_x, new_y, (max(edge_x) - min(edge_x)) * (max(edge_y) - min(edge_y))]

# Returns [overlap: bool, index1: int/None, index2: int/None] where indices are overlaping spheres
def overlap(x, y):  # Implement no-overlap constraint
    for x1, y1 in zip(x, y):
        i1 = x.index(x1)
        r1 = r[i1]
        for x2, y2 in zip(x, y):
            if x1 == x2 or y1 == y2:  # Skip same coords
                continue

            dist = math.dist([x1, y1], [x2, y2])  # Distance between circle1 and circle2
            i2 = x.index(x2)
            r2 = r[i2]
            if dist < (r1 + r2):
                return [True, i1, i2]  # Dist < combined radii so must be overlapping, return which circles are overlapping

    return [False, None, None]  # No overlap


def crude_monte_carlo(num_samples=5000):
    min = np.inf
    min_x_set = []
    min_y_set = []
    for i in range(num_samples):
        x = [get_rand_num(lower_bound, upper_bound) for i in range(N)]  # Generate set of x coords
        y = [get_rand_num(lower_bound, upper_bound) for i in range(N)]  # Generate set of y coords

        ex, ey, e = eval(x, y)

        if min > e:
            print(i, min)
            min = e
            min_x_set = ex
            min_y_set = ey
        
        if i % 10000 == 0:
            print(i)

    return min, min_x_set, min_y_set

min, x, y = crude_monte_carlo(1000000)
print(min)

fig, ax = plt.subplots()
for i in range(N):
    circ = plt.Circle((x[i], y[i]), r[i], color='b', fill=False)
    ax.add_patch(circ)
ax.set_xlim(-5 - max(r), 5 + max(r))
ax.set_ylim(-5 - max(r), 5 + max(r))
ax.set_aspect('equal')
plt.show()