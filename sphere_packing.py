__author__ = "Xinyue"
import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
import dccp

np.random.seed(0)
n = 10
r = np.linspace(1, 5, n)

c = cvx.Variable((n, 3))  # c is a [n] sets of 3D coordinates
constr = []
for i in range(n - 1):
    for j in range(i + 1, n):
        constr.append(cvx.norm(cvx.vec(c[i, :] - c[j, :]), 2) >= r[i] + r[j])  # The distance between 2 spheres (represented as 3D coords) must be greater than the radii of the two spheres combined
prob = cvx.Problem(cvx.Minimize(cvx.max(cvx.max(cvx.abs(c), axis=1) + r)), constr)  # Minimze the greatest distance from the origin
prob.solve(method="dccp", solver="ECOS", ep=1e-2, max_slack=1e-2)

l = cvx.max(cvx.max(cvx.abs(c), axis=1) + r).value * 2
pi = np.pi
ratio = pi * cvx.sum(cvx.square(r)).value / cvx.square(l).value
print("ratio =", ratio)
# plot
fig = plt.figure(figsize=(5, 5))
ax = fig.gca(projection='3d')
ax.set_aspect('auto')
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_border = [-l / 2, l / 2, l / 2, -l / 2, -l / 2]
y_border = [-l / 2, -l / 2, l / 2, l / 2, -l / 2]
for i in range(n):
    ax.plot_surface(
        c[i, 0].value + r[i] * np.cos(u) * np.sin(v), c[i, 1].value + r[i] * np.sin(u) * np.sin(v), c[i, 2].value + r[i] * np.cos(v), color="b"
    )
# plt.plot(x_border, y_border, "g")
# plt.axes().set_aspect("equal")
# plt.xlim([-l / 2, l / 2])
# plt.ylim([-l / 2, l / 2])
plt.show()