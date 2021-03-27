import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import dccp
import pandas as pd
from itertools import product
from scipy.stats import linregress
import pickle

# Seed so results are reproducible
np.random.seed(0)

####################
# Load in the data #
####################

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
Iij[Iij<1] = 1  # Fix really tiny intensities - 1 is still a really low intensity
Rij[Rij<1e-6] = 1e-6
Iij = Iij / 1000  # Scale intensities to kind of match distances

rad = []  # Create ordered list with radii to match the nodes in and intensities
for node in ids:
    rad.append(rdf.loc[rdf['ID'] == node]['Radius (mm)'].iloc[0])
    

##############################
# Setup optimization problem #
##############################
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

#########################
# Analyze dat data baby #
#########################

# Reload or save data to file so we don't have to run the optimization every time
load = True  # Set to false if you want to save a new run (i.e. you uncommented above)
if load:
    with open('save_c.file', 'rb') as f:
        c, dists = pickle.load(f)
else:
    with open('save_c.file', 'wb') as f:
        pickle.dump([c, dists], f)

# Plot 3D solution of optimized HY
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
# plt.show()

# rij_opt is the calculated optimum distance
rij_opt = np.zeros((n,n))

for i in range(n):
    for j in range(n):
        if i == j:
            continue
        # Calculate the distance between two spheres
        rij_opt[i][j] = cvx.norm(cvx.vec(c[i, :] - c[j, :]), 2).value

# Load in adjusted HY true distances
adj_real_dists = pd.read_csv('adjusted HY distances.csv', header=None).to_numpy()


# Reformat data so that we drop duplicates (should be 1/2 # of points because half of distance matrix is redundant)
print(linregress(rij.flatten(), rij_opt.flatten()))

# Make list of data points from below lower diagonal of distance matrices
trimmed_opt = rij_opt[np.tril_indices(rij_opt.shape[0], -1)]  # Non-redundant optimal solution
trimmed_real = adj_real_dists[np.tril_indices(adj_real_dists.shape[0], -1)]  # Non-redundant real model distances
trimmed_min = Rij[np.tril_indices(Rij.shape[0], -1)]  # Non-redundant minimum possible distances
trimmed_true = rij[np.tril_indices(rij.shape[0], -1)]  # Non-redundant true atlas distances

#
# Graph Adjusted True HY Dist vs Optimal Dist
#

fig, ax = plt.subplots()
ax.set_xlim(-1,7)
ax.set_ylim(-1,7)
ax.scatter(trimmed_real, trimmed_opt, color='slateblue', alpha=0.5)
plt.title('Adjusted HY Distances vs Optimal Solution', fontsize=18)
plt.xlabel(r'$Distance_{ij}$ (adjusted HY)', fontsize=18)
plt.ylabel(r'$Distance_{ij}$ ("optimal" solution)', fontsize=18)

lr = linregress(trimmed_real, trimmed_opt)
slope = round(lr.slope, 3)
intercept = round(lr.intercept, 3)
rvalue = round(lr.rvalue, 3)
pvalue = round(lr.pvalue, 3)
stderr = round(lr.stderr, 3)
x = np.linspace(-10, 10, 100)
y = slope * x + intercept
plt.plot(x, y, 'cadetblue', alpha=1)
plt.text(0.70, 0.8,
    "y = " + str(slope) + "x + " + str(intercept) +
    "\nr = " + str(rvalue) + "\np = " + str(pvalue),
    transform=ax.transAxes)

plt.show(block=False)

#
# Graph Minimum Possible Dist vs Optimal Dist
#
fig, ax = plt.subplots()
ax.set_xlim(0,2)
ax.set_ylim(0,3)
ax.scatter(trimmed_min, trimmed_opt, color='firebrick', alpha=0.5)
plt.title('Minimum Possible Model Distances vs Optimal Solution', fontsize=16)
plt.xlabel(r'$Distance_{ij}$ (minimum possible distance based on model)', fontsize=16)
plt.ylabel(r'$Distance_{ij}$ ("optimal" solution)', fontsize=16)

plt.show(block=False)

#
# Graph Unadjusted True HY Dist vs Optimal Dist
#
fig, ax = plt.subplots()
ax.set_xlim(-1,8)
ax.set_ylim(-1,8)
ax.scatter(trimmed_true, trimmed_opt, color='dodgerblue', alpha=0.5)
plt.title('True Atlas Distances vs Optimal Solution', fontsize=18)
plt.xlabel(r'$Distance_{ij}$ (true distance from atlas)', fontsize=18)
plt.ylabel(r'$Distance_{ij}$ ("optimal" solution)', fontsize=18)

lr = linregress(trimmed_true, trimmed_opt)
slope = round(lr.slope, 3)
intercept = round(lr.intercept, 3)
rvalue = round(lr.rvalue, 3)
pvalue = round(lr.pvalue, 3)
stderr = round(lr.stderr, 3)
x = np.linspace(-10, 10, 100)
y = slope * x + intercept
plt.plot(x, y, 'palevioletred', alpha=1)
plt.text(0.70, 0.8,
    "y = " + str(slope) + "x + " + str(intercept) +
    "\nr = " + str(rvalue) + "\np = " + str(pvalue),
    transform=ax.transAxes)

plt.show()

#######################################################################################
# How good is our model??? Check bounding box volume and efficiency (I hope it's good)#
#######################################################################################
c_max_x = 0
c_max_y = 0
c_max_z = 0
c_min_x = np.inf
c_min_y = np.inf
c_min_z = np.inf

# Find maximum distances in x, y, z directions to see how big our space used is
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

# Calculate efficiency of network
dist = rij_opt
dist[dist == 0] = 1e-10  # Just need to set these values to something non-zero so we can divide
Iij[Iij == 0.001] = 0  # Set self-loops back to zero so that the 1e-10 value up there doesn't matter
print('Optimized efficiency: %0.2f vs Actual Efficiency: %0.2f' % (np.sum((Iij / dist)), np.sum((Iij / rij))))
d2 = adj_real_dists
d2[d2 == 0] = 1e-6  # Same thing here
print('Optimized efficiency: %0.2f vs Adjusted Actual Efficiency: %0.2f' % (np.sum((Iij / dist)), np.sum((Iij / adj_real_dists))))