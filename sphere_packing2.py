import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import dccp

np.random.seed(0)
n = 10
r = np.linspace(1, 5, n)

c = cvx.Variable((n, 3))  # c is a [n] sets of 3D coordinates
dists = [[0 for x in range(n)] for y in range(n)]
intens = np.ones((10,10))
intens[3][8] = 40
intens[3][9] = 0.00001
#intens[6][9] = 1000
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
prob = cvx.Problem(cvx.Minimize(cvx.multiply(cvx.norm(cvx.abs(d), 1), len(dists) * len(dists))), constr)  # Minimize the greatest distance from the origin
prob.solve(method="dccp", solver="ECOS", ep=1e-2, max_slack=1e-2)
#print(d.value)

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
def rotate(angle):
    ax.view_init(azim=angle)
rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,362,2),interval=100)
rot_animation.save('./rotation2.gif', dpi=80, writer='imagemagick')
plt.show()