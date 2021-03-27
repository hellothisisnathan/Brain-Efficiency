import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import dccp
import pandas as pd

np.random.seed(0)
n = 10
r = np.linspace(1, 5, n)

c = cvx.Variable((n, 3))  # c is a [n] sets of 3D coordinates
dists = [[0 for x in range(n)] for y in range(n)]
intens = np.ones((10,10))
intens[3][9] = 10
intens[3][8] = 100
# intens[3][0] = 50
intens[6][9] = 1
constr = []
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        # Calculate the distance between two spheres
        dists[i][j] = cvx.norm(cvx.vec(c[i, :] - c[j, :]), 2)
        # The distance between 2 spheres (represented as 3D coords) must be greater than the radii of the two spheres combined
        constr.append(dists[i][j] >= r[i] + r[j])
        #print("%i, %i: dist= >= %f" % (i, j, r[i] + r[j]))
d = cvx.multiply(cvx.bmat(dists), intens)
#prob = cvx.Problem(cvx.Minimize(cvx.norm(cvx.abs(d), 1)), constr)  # Minimize the greatest distance from the origin
# prob = cvx.Problem(cvx.Minimize(cvx.multiply(cvx.norm(cvx.abs(d), 1), 1 / (len(dists) * len(dists)))), constr)  # Minimize the greatest distance from the origin
# prob.solve(method="dccp", solver="MOSEK", ep=1e-2, max_slack=1e-2)
# print(pd.DataFrame(d.value))

# # plot
# fig = plt.figure(figsize=(5, 5))
# ax = fig.gca(projection='3d')
# ax.set_aspect('auto')
# u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
# for i in range(n):
#     colors = "b"
#     if i == 3:
#         colors = "r"
#     elif i == 9:
#         colors = "g"
#     elif i == 8:
#         colors = "y"
#     else:
#         colors = "b"
#     ax.plot_surface(
#         c[i, 0].value + r[i] * np.cos(u) * np.sin(v), c[i, 1].value + r[i] * np.sin(u) * np.sin(v), c[i, 2].value + r[i] * np.cos(v), color=colors
#     )
# # def rotate(angle):
# #     ax.view_init(azim=angle)
# # rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,362,2),interval=100)
# # rot_animation.save('./rotation2.gif', dpi=80, writer='imagemagick')
# c_max_x = 0
# c_max_y = 0
# c_max_z = 0

# # for v in c.value:
# #     if abs(v[0]) > c_max_x:
# #         c_max_x = abs(v[0])
# #     if abs(v[1]) > c_max_y:
# #         c_max_y = abs(v[1])
# #     if abs(v[2]) > c_max_z:
# #         c_max_z = abs(v[2])

# for i in range (n):
#     if abs(c[i].value[0]) + r[i] > c_max_x:
#         c_max_x = abs(c[i].value[0]) + r[i]
#     if abs(c[i].value[1]) + r[i] > c_max_y:
#         c_max_y = abs(c[i].value[1]) + r[i]
#     if abs(c[i].value[2]) + r[i] > c_max_z:
#         c_max_z = abs(c[i].value[2]) + r[i]
# print('Brain bounded by prism %0.2f x %0.2f x %0.2f ' % (c_max_x, c_max_y, c_max_z))
# print('Vol = ', c_max_x * c_max_y * c_max_z)

# plt.show()

c = cvx.Variable((n, 3))
constr = []
dists_org = [[0 for x in range(n)] for y in range(n)]

for i in range(n):
    for j in range(n):
        if i == j:
            continue
        # Calculate the distance between two spheres
        dists[i][j] = cvx.norm(cvx.vec(c[i, :] - c[j, :]), 2)
        # dists_org[i][j] = cvx.norm(cvx.vec(c[i, :] - [0,0,0]), 2)
        # The distance between 2 spheres (represented as 3D coords) must be greater than the radii of the two spheres combined
        constr.append(dists[i][j] >= r[i] + r[j])
        #print("%i, %i: dist= >= %f" % (i, j, r[i] + r[j]))
prob = cvx.Problem(cvx.Minimize(cvx.sum(cvx.multiply(cvx.bmat(dists), intens))), constr)
prob.solve(method="dccp", solver="ECOS", ep=1e-2, max_slack=1e-2)
# import pandas as pd
# print(pd.DataFrame(c.value))
# print(pd.DataFrame(cvx.bmat(dists).value))
# print(pd.DataFrame(cvx.multiply(cvx.bmat(dists), intens).value))
# print(cvx.sum(cvx.multiply(cvx.bmat(dists), intens)).value)
# print(cvx.sum(cvx.bmat(dists)).value)


# plot
fig = plt.figure(figsize=(5, 5))
ax = fig.gca(projection='3d')
ax.set_aspect('auto')
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
for i in range(n):
    colors = "b"
    if i == 3:
        colors = "r"
    elif i == 9:
        colors = "g"
    elif i == 8:
        colors = "y"
    else:
        colors = "b"
    ax.plot_surface(
        c[i, 0].value + r[i] * np.cos(u) * np.sin(v), c[i, 1].value + r[i] * np.sin(u) * np.sin(v), c[i, 2].value + r[i] * np.cos(v), color=colors
    )
# def rotate(angle):
#     ax.view_init(azim=angle)
# rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,362,2),interval=100)
# rot_animation.save('./rotation2.gif', dpi=80, writer='imagemagick')
c_max_x = 0
c_max_y = 0
c_max_z = 0
c_min_x = np.inf
c_min_y = np.inf
c_min_z = np.inf

# for v in c.value:
#     if abs(v[0]) > c_max_x:
#         c_max_x = abs(v[0])
#     if abs(v[1]) > c_max_y:
#         c_max_y = abs(v[1])
#     if abs(v[2]) > c_max_z:
#         c_max_z = abs(v[2])

for i in range (n):
    if c[i].value[0] + r[i] > c_max_x:
        c_max_x = c[i].value[0] + r[i]
    if c[i].value[1] + r[i] > c_max_y:
        c_max_y = c[i].value[1] + r[i]
    if c[i].value[2] + r[i] > c_max_z:
        c_max_z = c[i].value[2] + r[i]
    if c[i].value[0] - r[i] < c_min_x:
        c_min_x = c[i].value[0] - r[i]
    if c[i].value[1] - r[i] < c_min_y:
        c_min_y = c[i].value[1] - r[i]
    if c[i].value[2] - r[i] < c_min_z:
        c_min_z = c[i].value[2] - r[i]
print('Brain bounded by box %0.2f x %0.2f x %0.2f ' % (abs(c_max_x - c_min_x), abs(c_max_y - c_min_y), abs(c_max_z - c_min_z)))
print('Vol = ', abs(c_max_x - c_min_x) * abs(c_max_y - c_min_y) * abs(c_max_z - c_min_z))
dist =  cvx.bmat(dists).value
dist[dist == 0] = 1
print('Sum of Efficiency = %0.2f' % np.sum((intens / dist)))
# print(pd.DataFrame(intens/dist))

plt.show()