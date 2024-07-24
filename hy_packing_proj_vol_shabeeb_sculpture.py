import cvxpy as cvx
import numpy as np
import matplotlib as mpl
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

df = pd.read_csv('./hy edge list intensities and projection volumes.csv')  # Edge list with distances and r1+r2 columns
rdf = pd.read_csv('../all brain volumes.csv')  # CSV with radius info for nodes

nodes = np.unique(df['n1'].tolist() + df['n2'].tolist())
n = len(nodes)

## define the matrix of distances between nodes. It is symmetric
rij = np.zeros((n,n))  # Distance matrix
rij[rij<1.e-6] = 1.e-6
Rij = np.zeros((n,n))  # R1+R2 combined radii (min distance) matrix
Iij = np.zeros((n,n))  # projection volume matrix
iij = np.zeros((n,n))  # intensity matrix

ids = [0 for x in range(n)]  # List to lookup ids later

# Create distance, r1+r2, projection volume matrices
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
            Iij[i1,i2] = tempdf['projection_volume']
            iij[i1,i2] = tempdf['intensity']
            ids[i1] = tempdf['n1'].iloc[0]
## symmetrize the tensors
# rij = (rij + rij.transpose())/2.  # Distance matrix		
# Rij = (Rij + Rij.transpose())/2.  # R1+R2 combined radii (min distance) matrix
# Iij = (Iij + Iij.transpose())/2.  # projection volume matrix
# iij = (iij + iij.transpose())/2.  # intensity matrix
Iij[Iij<0.0001] = 0.0001  # Fix really tiny projection volumes - 0.0001 is still a really low projection volume
Rij[Rij<1e-6] = 1e-6
#Iij = Iij / 1000  # Scale projection volumes to kind of match distances

rad = []  # Create ordered list with radii to match the nodes in and projection volumes
names = []  # Create ordered list with names to match ids
for node in ids:
    rad.append(rdf.loc[rdf['ID'] == node]['Radius (mm)'].iloc[0])
    names.append(rdf.loc[rdf['ID'] == node]['Name'].iloc[0])
    

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
# prob = cvx.Problem(cvx.Minimize(cvx.sum(cvx.multiply(cvx.bmat(dists), Iij))), constr)  # Minimize the value (dists) * (projection volumes)... Should be the same minimization as projection volumes/dist?
# prob.solve(method="dccp", solver="ECOS", ep=1e-2, max_slack=1e-2)
#################################################################################

#########################
# Analyze dat data baby #
#########################

# efficiency=0
# for q in range(0,len(rij)):
#     for r in range(0,len(rij)):
#         if r==q: continue#avoid divide by zero!
#         elif Iij[q,r]<1e-10: continue #cutoff noise projections
#         efficiency+=Iij[q,r]/rij[q,r]

# print(efficiency)
# exit()

# Reload or save data to file so we don't have to run the optimization every time
load = True  # Set to false if you want to save a new run (i.e. you uncommented above)
if load:
    with open('save_c_pv.file', 'rb') as f:
        c, dists = pickle.load(f)
else:
    with open('save_c_pv.file', 'wb') as f:
        pickle.dump([c, dists], f)

# Plot 3D solution of optimized HY
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.set_aspect('auto')
ax.set_title('Optimized Hypothalmus', fontsize=18)
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
cmap = plt.cm.get_cmap('hsv', n)
for i in range(n):
    ax.plot_surface(
        c[i, 0].value + rad[i] * np.cos(u) * np.sin(v), c[i, 1].value + rad[i] * np.sin(u) * np.sin(v), c[i, 2].value + rad[i] * np.cos(v), color=cmap(i)
    )
ax.set_xlabel('Displacement (cm)', fontsize=16)
ax.set_ylabel('\nDisplacement (cm)', fontsize=16)
ax.set_zlabel('Displacement (cm)', fontsize=16)
ax.set_xlim([-3,3])
ax.set_ylim([-3,3])
ax.set_zlim([-3,3])
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.tick_params(axis='z', labelsize=16)

# Add legend (jank)
ax = fig.add_subplot(1,2,2)
lines = []
for i in range(n):
    lines.append(mpl.lines.Line2D([10],[0], linestyle="none", c=cmap(i), marker = 'o'))
ax.legend(lines, names, numpoints = 1)
ax.axis('off')
# def rotate(angle):
#     ax.view_init(azim=angle)
# rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,362,2),interval=100)
#rot_animation.save('./results/animations/optimized_hy_animation.gif', dpi=160, writer='imagemagick')
# plt.savefig('./results/optimized_hy_render_proj_vol.png', dpi=300)
plt.show()

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

#
# Reformat data so that we drop duplicates (should be 1/2 # of points because half of distance matrix is redundant)
#
# Make list of data points from below lower diagonal of distance matrices
trimmed_opt = rij_opt[np.tril_indices(rij_opt.shape[0], -1)]  # Non-redundant optimal solution
trimmed_real = adj_real_dists[np.tril_indices(adj_real_dists.shape[0], -1)]  # Non-redundant real model distances
trimmed_min = Rij[np.tril_indices(Rij.shape[0], -1)]  # Non-redundant minimum possible distances
trimmed_true = rij[np.tril_indices(rij.shape[0], -1)]  # Non-redundant true atlas distances

#
# Intensity vs Projection Volume
#

fig, ax = plt.subplots()
ax.set_xlim(-0.1,1)
ax.set_ylim(-1000,16000)
ax.scatter(Iij.flatten(), iij.flatten(), color='slateblue', alpha=0.5)
plt.title('Intensity vs Projection Volume', fontsize=16)
plt.xlabel('Intensity', fontsize=14)
plt.ylabel('Projection Volume', fontsize=14)

lr = linregress(Iij.flatten(), iij.flatten())
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

# plt.savefig('./results/intens_vs_projvol.png', dpi=300)
plt.show(block=False)
# pd.DataFrame({'real':trimmed_real, 'opt': trimmed_opt}).to_csv('./results/csv/real_vs_opt.csv', index=False)

#
# Graph Adjusted True HY Dist vs Optimal Dist
#

fig, ax = plt.subplots()
ax.set_xlim(0,7)
ax.set_ylim(0,7)
ax.scatter(trimmed_real, trimmed_opt, color='slateblue', alpha=0.5)
plt.title('Adjusted HY Distances vs Optimal Solution', fontsize=16)
plt.xlabel(r'$Distance_{ij}$ (adjusted HY)', fontsize=14)
plt.ylabel(r'$Distance_{ij}$ ("optimal" solution)', fontsize=14)

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

# plt.savefig('./results/adjusted_vs_opt.png', dpi=300)
plt.show(block=False)
# pd.DataFrame({'real':trimmed_real, 'opt': trimmed_opt}).to_csv('./results/csv/real_vs_opt.csv', index=False)

#
# Graph Minimum Possible Dist vs Optimal Dist
#
fig, ax = plt.subplots()
ax.set_xlim(0,2)
ax.set_ylim(0,3)
ax.scatter(trimmed_min, trimmed_opt, color='firebrick', alpha=0.5)
plt.title('Minimum Possible Model Distances vs Optimal Solution', fontsize=16)
plt.xlabel(r'$Distance_{ij}$ (minimum possible distance based on model)', fontsize=14)
plt.ylabel(r'$Distance_{ij}$ ("optimal" solution)', fontsize=14)

# plt.savefig('./results/min_vs_opt.png', dpi=300)
plt.show(block=False)

#
# Graph Unadjusted True HY Dist vs Optimal Dist
#
fig, ax = plt.subplots()
ax.set_xlim(0,7)
ax.set_ylim(0,7)
ax.scatter(trimmed_true, trimmed_opt, color='dodgerblue', alpha=0.5)
plt.title('True Atlas Distances vs Optimal Solution', fontsize=16)
plt.xlabel(r'$Distance_{ij}$ (true distance from atlas)', fontsize=14)
plt.ylabel(r'$Distance_{ij}$ ("optimal" solution)', fontsize=14)

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

# plt.savefig('./results/true_vs_opt.png', dpi=300)
plt.show()
# pd.DataFrame({'true': trimmed_true, 'opt': trimmed_opt}).to_csv('./results/csv/true_vs_opt.csv', index=False)

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
print('#' * 70)
print('Hypothalamus bounded by box %0.2f x %0.2f x %0.2f ' % (abs(c_max_x - c_min_x), abs(c_max_y - c_min_y), abs(c_max_z - c_min_z)))
print('Vol = ', abs(c_max_x - c_min_x) * abs(c_max_y - c_min_y) * abs(c_max_z - c_min_z))

# Load in original HY coords to compare bounding box and volumes
true_coords = pd.read_csv('true HY coordinates.csv', header=None).to_numpy()
true_max_x = 0
true_max_y = 0
true_max_z = 0
true_min_x = np.inf
true_min_y = np.inf
true_min_z = np.inf
for i in range (n):
    if true_coords[i][0] + rad[i] > true_max_x:
        true_max_x = true_coords[i][0] + rad[i]
    if true_coords[i][1] + rad[i] > true_max_y:
        true_max_y = true_coords[i][1] + rad[i]
    if true_coords[i][2] + rad[i] > true_max_z:
        true_max_z = true_coords[i][2] + rad[i]
    if true_coords[i][0] - rad[i] < true_min_x:
        true_min_x = true_coords[i][0] - rad[i]
    if true_coords[i][1] - rad[i] < true_min_y:
        true_min_y = true_coords[i][1] - rad[i]
    if true_coords[i][2] - rad[i] < true_min_z:
        true_min_z = true_coords[i][2] - rad[i]
adj_coords = pd.read_csv('adjusted HY coordinates.csv', header=None).to_numpy()
adj_max_x = 0
adj_max_y = 0
adj_max_z = 0
adj_min_x = np.inf
adj_min_y = np.inf
adj_min_z = np.inf
for i in range (n):
    if adj_coords[i][0] + rad[i] > adj_max_x:
        adj_max_x = adj_coords[i][0] + rad[i]
    if adj_coords[i][1] + rad[i] > adj_max_y:
        adj_max_y = adj_coords[i][1] + rad[i]
    if adj_coords[i][2] + rad[i] > adj_max_z:
        adj_max_z = adj_coords[i][2] + rad[i]
    if adj_coords[i][0] - rad[i] < adj_min_x:
        adj_min_x = adj_coords[i][0] - rad[i]
    if adj_coords[i][1] - rad[i] < adj_min_y:
        adj_min_y = adj_coords[i][1] - rad[i]
    if adj_coords[i][2] - rad[i] < adj_min_z:
        adj_min_z = adj_coords[i][2] - rad[i]

print('-' * 70)
print('True HY bounded by box %0.2f x %0.2f x %0.2f ' % (abs(true_max_x - true_min_x), abs(true_max_y - true_min_y), abs(true_max_z - true_min_z)))
print('True Vol = ', abs(true_max_x - true_min_x) * abs(true_max_y - true_min_y) * abs(true_max_z - true_min_z))
print('Adjusted HY bounded by box %0.2f x %0.2f x %0.2f ' % (abs(adj_max_x - adj_min_x), abs(adj_max_y - adj_min_y), abs(adj_max_z - adj_min_z)))
print('Adjusted Vol = ', abs(adj_max_x - adj_min_x) * abs(adj_max_y - adj_min_y) * abs(adj_max_z - adj_min_z))
print('-' * 70)

###################################
# Calculate efficiency of network #
###################################
dist = rij_opt
dist[dist == 0] = 1e-10  # Just need to set these values to something non-zero so we can divide
Iij[Iij == 0.0001] = 0  # Set self-loops back to zero so that the 1e-10 value up there doesn't matter
print('Optimized efficiency: %0.2f vs Actual Efficiency: %0.2f' % (np.sum((Iij / dist)), np.sum((Iij / rij))))
d2 = adj_real_dists
d2[d2 == 0] = 1e-10  # Same thing here
print('Optimized efficiency: %0.2f vs Adjusted Actual Efficiency: %0.2f' % (np.sum((Iij / dist)), np.sum((Iij / adj_real_dists))))
print('#' * 70)