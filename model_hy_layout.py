import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from matplotlib import rc
rc('font', **{
    'family': 'serif',
    'style': 'normal',
    'weight': 'medium',
    'serif': 'arial',
})

mcc = MouseConnectivityCache()
structure_tree = mcc.get_structure_tree()

# Seed so results are reproducible
np.random.seed(0)

df = pd.read_csv('../edge_withDistancesAndRadii.csv')  # Edge list with distances and r1+r2 columns
rdf = pd.read_csv('../all brain volumes.csv')  # CSV with radius info for nodes
real_coords = pd.read_csv('./true HY coordinates.csv', header=None).to_numpy()

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
names = []  # Create ordered list with names to match ids
abbrev = []  # Create ordered list w/ abbrev to match ids
for node in ids:
    rad.append(rdf.loc[rdf['ID'] == node]['Radius (mm)'].iloc[0])
    names.append(rdf.loc[rdf['ID'] == node]['Name'].iloc[0])
    abbrev.append(structure_tree.get_structures_by_id([node])[0]['acronym'])

print(rad)
exit()

# real_coords = np.array([
#     [-2.46, 0.6, 5.65],
#     [-0.58, 3.4, 4.94],
#     [-0.94, 0.25, 4.8],
#     [-4.36, 0.3, 4.55],
#     [-0.58, 0.2, 4.75],
#     [-2.3, 0.25, 5.85],
#     [-1.34, 1, 4.84],
#     [-2.46, 1, 5.3],
#     [-1.46, 0.1, 5.9],
#     [0.5, 0.81, 4.85],
#     [0.98, 0.25, 4.8],
#     [-0.34, 0.2, 5.6],
#     [-4.36, 0.5, 1],
#     [-2.18, 1.25, 4.95],
#     [-0.46, 1.15, 5.45],
#     [-1.34, 1.75, 4.5],
#     [-2.7, 0.13, 5.2],
#     [0.02, 0.25, 4.9],
#     [0.38, 0.7, 5],
#     [-2.35, 0.13, 4.98],
#     [-0.7, 1.13, 5.23],
#     [-1.35, -0.75, 5.75],
#     [-2.18, 1.75, 4.12],
#     [-1.5, 0.35, 5.2],
#     [-2.3, 0.25, 4.37],
#     [-2.46, 0.55, 5.65],
# ])

# adjust coords
for i in range(np.shape(real_coords)[0]):
    real_coords[i][0] = real_coords[i][0] + 1.5
    real_coords[i][1] = real_coords[i][1] - 1
    real_coords[i][2] = real_coords[i][2] - 4.5

# Save coords to file
# pd.DataFrame(real_coords).to_csv('true HY coordinates.csv', index=False, header=False)
# Plot original HY
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.set_aspect('auto')

font = {
    'family': 'serif',
    'size': 18,
    'style': 'normal',
    'weight': 'medium',
    'fontname': 'arial',
}
ax.set_title('Original HY Layout', fontdict=font, y=1.05)
ax.set_xlabel('X (cm)', fontdict=font)
ax.set_ylabel('Y (cm)', fontdict=font)
ax.set_zlabel('Z (cm)', fontdict=font)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
# cmap = plt.cm.get_cmap('hsv', n)
cmap = [
    'black',
    'gray',
    'lightcoral',
    'red',
    'tomato',
    'chocolate',
    'darkorange',
    'gold',
    'yellowgreen',
    'greenyellow',
    'forestgreen',
    'lime',
    'mediumseagreen',
    'mediumspringgreen',
    'turquoise',
    'darkcyan',
    'deepskyblue',
    'skyblue',
    'midnightblue',
    'slateblue',
    'blueviolet',
    'purple',
    'fuchsia',
    'mediumvioletred',
    'hotpink',
    'crimson',
]

for i in range(n):
    ax.plot_surface(
        real_coords[i, 0] + rad[i] * np.cos(u) * np.sin(v), real_coords[i, 1] + rad[i] * np.sin(u) * np.sin(v), real_coords[i, 2] + rad[i] * np.cos(v), color=cmap[i]
    )
ax.view_init(36, 46)
ax = fig.add_subplot(1,2,2)
# Add legend (jank)
lines = []
for i in range(n):
    lines.append(mpl.lines.Line2D([10],[0], linestyle="none", c=cmap[i], marker = 'o'))
ax.legend(lines, abbrev, numpoints = 1)
ax.axis('off')

# plt.savefig('./results/updated figs 11-19-21/original_hy_layout.png', dpi=300)
plt.show()

###
# Next graph same thing with outliers removed
###
outliers = ['PVH', 'ADP', 'SBPV']

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.set_aspect('auto')

font = {
    'family': 'serif',
    'size': 18,
    'style': 'normal',
    'weight': 'medium',
    'fontname': 'arial',
}
ax.set_title('HY Layout - Outliers Removed', fontdict=font, y=1.05)
ax.set_xlabel('X (cm)', fontdict=font)
ax.set_ylabel('Y (cm)', fontdict=font)
ax.set_zlabel('Z (cm)', fontdict=font)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
# cmap = plt.cm.get_cmap('hsv', n)
cmap = [
    'black',
    'gray',
    'lightcoral',
    'red',
    'tomato',
    'chocolate',
    'darkorange',
    'gold',
    'yellowgreen',
    'greenyellow',
    'forestgreen',
    'lime',
    'mediumseagreen',
    'mediumspringgreen',
    'turquoise',
    'darkcyan',
    'deepskyblue',
    'skyblue',
    'midnightblue',
    'slateblue',
    'blueviolet',
    'purple',
    'fuchsia',
    'mediumvioletred',
    'hotpink',
    'crimson',
]

for i in range(n):
    if abbrev[i] not in outliers:
        ax.plot_surface(
            real_coords[i, 0] + rad[i] * np.cos(u) * np.sin(v), real_coords[i, 1] + rad[i] * np.sin(u) * np.sin(v), real_coords[i, 2] + rad[i] * np.cos(v), color=cmap[i]
        )
ax.view_init(36, 46)
ax = fig.add_subplot(1,2,2)
# Add legend (jank)
lines = []
for i in range(n):
    lines.append(mpl.lines.Line2D([10],[0], linestyle="none", c=cmap[i], marker = 'o'))
ax.legend(lines, abbrev, numpoints = 1)
ax.axis('off')

# plt.savefig('./results/updated figs 11-19-21/no_outliers_hy_layout.png', dpi=300)
plt.show()
exit()



def overlaps(coords1, coords2, r1, r2):
    return np.linalg.norm(coords1 - coords2) - r1 - r2 < 0

nudged_coords = real_coords
laps = True
def check():
    laps = False
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if overlaps(nudged_coords[i], nudged_coords[j], rad[i], rad[j]):
                laps = True
                nudged_coords[i, 0] += np.random.normal(0, 0.05)
                nudged_coords[i, 1] += np.random.normal(0, 0.05)
                nudged_coords[i, 2] += np.random.normal(0, 0.05)
                print('overlap between %i and %i' % (i,j))
    return laps

while laps:
    laps = False
    laps = check()

new_dists = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        new_dists[i,j] = np.linalg.norm(nudged_coords[i] - nudged_coords[j])
# print('old dists', pd.DataFrame(rij))
# print('new dists', pd.DataFrame(new_dists))
print('average nudged', np.average(new_dists - rij))

# Plot HY corrected for model
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.set_aspect('auto')
ax.set_title('Adjusted HY Layout', fontsize=18)
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
cmap = plt.cm.get_cmap('hsv', n)
for i in range(n):
    ax.plot_surface(
        nudged_coords[i, 0] + rad[i] * np.cos(u) * np.sin(v), nudged_coords[i, 1] + rad[i] * np.sin(u) * np.sin(v), nudged_coords[i, 2] + rad[i] * np.cos(v), color=cmap(i)
    )
ax = fig.add_subplot(1,2,2)
# Add legend (jank)
lines = []
for i in range(n):
    lines.append(mpl.lines.Line2D([10],[0], linestyle="none", c=cmap(i), marker = 'o'))
ax.legend(lines, names, numpoints = 1)
ax.axis('off')

# plt.savefig('./results/adjusted_hy_layout.png', dpi=300)
plt.show()

# pd.DataFrame(nudged_coords).to_csv('adjusted HY coordinates.csv', index=False, header=False)
# pd.DataFrame(new_dists).to_csv('adjusted HY distances.csv', index=False, header=False)