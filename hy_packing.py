import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import dccp
import pandas as pd
from itertools import product

# Seed so results are reproducible
np.random.seed(0)

df = pd.read_csv('../edge_withDistancesAndRadii.csv')  # Edge list with distances and r1+r2 columns
rdf = pd.read_csv('../all brain volumes.csv')  # CSV with radius info for nodes

nodes = np.unique(df['n1'].tolist() + df['n2'].tolist())
n = len(nodes)

## define the matrix of distances between nodes. It is symmetric
rij = np.zeros((n,n))  # Distance matrix
rij[rij<1.e-6] = 1.e-6
Rij = np.zeros((n,n))  # R1+R2 combined radii (min distance) matrix
Iij = np.zeros((n,n))  # Intensity matrix

ids = [0 for x in range(n)]  # List to lookup ids later

# Create distance, r1+r2, intensity matrices
for i1 in range(n):
    for i2 in range(n):
        if i1 == i2: continue
        tempdf = df[(df['n1']==nodes[i1])&(df['n2']==nodes[i2])]
        if tempdf.index.size > 1:
            print("two edges for (%s,%s). only one will be stored. drop the duplicate and run again"%(nodes[i1],nodes[i2]))
            #raise(Exception("please drup the duplicate and run again"))
        elif tempdf.index.size == 1:
            rij[i1,i2] = tempdf['distance']
            Rij[i1,i2] = tempdf['R1+R2']
            Iij[i1,i2] = tempdf['intensity']
            ids[i1] = tempdf['n1'].iloc[0]
## symmetrize the tensors
rij = (rij + rij.transpose())/2.  # Distance matrix		
Rij = (Rij + Rij.transpose())/2.  # R1+R2 combined radii (min distance) matrix
Iij = (Iij + Iij.transpose())/2.  # Intensity matrix
Iij[Iij<1] = 1
Rij[Rij<1e-6] = 1e-6
Iij = Iij / 1000  # Scale intensities to kind of match distances

rad = []  # Create ordered list with radii to match the nodes in and intensities
for node in ids:
    rad.append(rdf.loc[rdf['ID'] == node]['Radius (mm)'].iloc[0])

c = cvx.Variable((n, 3))  # c is [n] sets of 3D coordinates
dists = [[0 for x in range(n)] for y in range(n)]  # Generate 26x26 array to hold variable calculations for distance constraints
constr = []  # Constraints list
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        # Calculate the distance between two spheres
        dists[i][j] = cvx.norm(cvx.vec(c[i, :] - c[j, :]), 2)
        # The distance between 2 spheres (represented as 3D coords) must be greater than the radii of the two spheres combined
        constr.append(dists[i][j] >= rad[i] + rad[j])
        #print("%i, %i: dist= >= %f" % (i, j, r[i] + r[j]))

#################################################################################
# Problem solver - takes a long time!
# prob = cvx.Problem(cvx.Minimize(cvx.sum(cvx.multiply(cvx.bmat(dists), Iij))), constr)  # Minimize the value (dists) * (intensities)... Should be the same minimization as intensities/dist?
# prob.solve(method="dccp", solver="ECOS", ep=1e-2, max_slack=1e-2)
#################################################################################

import pickle
# with open('save_c.file', 'wb') as f:
#     pickle.dump([c, dists], f)
with open('save_c.file', 'rb') as f:
    c, dists = pickle.load(f)

# plot
fig = plt.figure(figsize=(20, 20))
ax = fig.gca(projection='3d')
ax.set_aspect('auto')
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
cmap = plt.cm.get_cmap('hsv', n)
for i in range(n):
    ax.plot_surface(
        c[i, 0].value + rad[i] * np.cos(u) * np.sin(v), c[i, 1].value + rad[i] * np.sin(u) * np.sin(v), c[i, 2].value + rad[i] * np.cos(v), color=cmap(i)
    )
# def rotate(angle):
#     ax.view_init(azim=angle)
# rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,362,2),interval=100)
#rot_animation.save('./opt_hy.gif', dpi=80, writer='imagemagick')
plt.show()

rij_opt = np.zeros((n,n))

for i in range(n):
    for j in range(n):
        if i == j:
            continue
        # Calculate the distance between two spheres
        rij_opt[i][j] = cvx.norm(cvx.vec(c[i, :] - c[j, :]), 2).value


fig, ax = plt.subplots()
# ax.set_xlim(0,6)
# ax.set_ylim(0,6)
ax.scatter(Rij.flatten(), rij_opt.flatten(), alpha=0.2)
plt.xlabel(r'$Distance_{ij}$ (minimum possible distance based on model)', fontsize=18)
plt.ylabel(r'$Distance_{ij}$ ("optimal" solution)', fontsize=18)
# plt.show()

fig, ax = plt.subplots()
ax.set_xlim(-1,8)
ax.set_ylim(-1,8)
ax.scatter(rij.flatten(), rij_opt.flatten(), alpha=0.2)
plt.xlabel(r'$Distance_{ij}$ (true)', fontsize=18)
plt.ylabel(r'$Distance_{ij}$ ("optimal" solution)', fontsize=18)
# plt.show()

fig, ax = plt.subplots()
ax.scatter(range(n*n), Rij.flatten() - rij_opt.flatten(), color='k', alpha=0.4)
# plt.show()

r = rij[0,1]/rij_opt[0,1]
for i, j in product(range(n),range(n)):
	if i>=j: continue
	print("d[%d,%d]: %1.2f(true) %1.2f(sol.)"%(i,j,rij[i,j], rij_opt[i,j]*r))

df['opt. distance'] = df.index.size*[0]
for i1 in range(n):
	for i2 in range(n):
		if i1 == i2: continue
		tempdf = df[(df['n1']==nodes[i1])&(df['n2']==nodes[i2])]
		if tempdf.index.size > 1:
			print("two edges for (%s,%s). only one will be stored. drop the duplicate and run again"%(nodes[i1],nodes[i2]))
			#raise(Exception("please drup the duplicate and run again"))
		elif tempdf.index.size == 1:
			df['opt. distance'][(df['n1']==nodes[i1])&(df['n2']==nodes[i2])] = r*rij_opt[i1,i2]

	
	
fig, ax = plt.subplots()
ax.scatter(df['distance'],df['opt. distance'],alpha=0.2)
ax.set_xlim(0,8)
ax.set_ylim(0,8)
# ax.set_xlabel(r'$r_{ij}$ (true)')
# ax.set_ylabel(r'$r_{ij}$ (optimal solution)')
plt.xlabel(r'$Distance_{ij}$ (true)', fontsize=18)
plt.ylabel(r'$Distance_{ij}$ (optimal solution)', fontsize=18)
plt.title('Optimal Model Arrangement vs. True Distance', fontsize=18)
# plt.show()
# fig.savefig('rij_true_vs_OptimalSolution.png')	
	
	
	
fig, ax = plt.subplots()
ax.scatter(df['opt. distance']-df['distance'],df['intensity'],color='k',alpha=0.4)
ax.set_xlabel(r'$\Delta r_{ij}$')
ax.set_ylabel(r'intensity')
# plt.show()

c_max_x = 0
c_max_y = 0
c_max_z = 0
c_min_x = np.inf
c_min_y = np.inf
c_min_z = np.inf

a = rad[i]
for i in range (n):
    if c[i].value[0] + rad[i] > c_max_x:
        c_max_x = c[i].value[0] + rad[i]
    if c[i].value[1] + rad[i] > c_max_y:
        c_max_y = c[i].value[1] + rad[i]
    if c[i].value[2] + rad[i] > c_max_z:
        c_max_z = c[i].value[2] + rad[i]
    if c[i].value[0] - rad[i] < c_min_x:
        c_min_x = c[i].value[0] - rad[i]
    if c[i].value[1] - rad[i] < c_min_y:
        c_min_y = c[i].value[1] - rad[i]
    if c[i].value[2] - rad[i] < c_min_z:
        c_min_z = c[i].value[2] - rad[i]
print('Brain bounded by box %0.2f x %0.2f x %0.2f ' % (abs(c_max_x - c_min_x), abs(c_max_y - c_min_y), abs(c_max_z - c_min_z)))
print('Vol = ', abs(c_max_x - c_min_x) * abs(c_max_y - c_min_y) * abs(c_max_z - c_min_z))
dist = cvx.bmat(dists).value
dist[dist == 0] = 1
print('Optimized efficiency: %0.2f vs Actual Efficiency: %0.2f' % (np.sum((Iij / dist)), np.sum((Iij / rij))))