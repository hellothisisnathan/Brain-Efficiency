import numpy as np
import pandas as pd
import math
import time
import random
from matplotlib import pyplot as plt
from IPython.display import clear_output

####################
# Load in the data #
####################

df = pd.read_csv('../../edge_withDistancesAndRadii.csv')  # Edge list with distances and r1+r2 columns
rdf = pd.read_csv('../../all brain volumes.csv')  # CSV with radius info for nodes

nodes = np.unique(df['n1'].tolist() + df['n2'].tolist())  # Node list
N = len(nodes)  # # nodes
intens = np.zeros((N,N))  # Intensity matrix
ids = [0 for x in range(N)]  # List to lookup ids later

# Create distance, r1+r2, intensity matrices
for i1 in range(N):
    for i2 in range(N):
        if i1 == i2: continue
        tempdf = df[(df['n1']==nodes[i1])&(df['n2']==nodes[i2])]
        if tempdf.index.size > 1:
            print("two edges for (%s,%s). only one will be stored. drop the duplicate and run again"%(nodes[i1],nodes[i2]))
            #raise(Exception("please drup the duplicate and run again"))
        elif tempdf.index.size == 1:
            intens[i1,i2] = tempdf['intensity']
            ids[i1] = tempdf['n1'].iloc[0]
## symmetrize the tensors
intens = (intens + intens.transpose())/2.  # Intensity matrix
intens[intens<1] = 1  # Fix really tiny intensities - 1 is still a really low intensity
intens = intens / 1000  # Scale intensities to kind of match distances

r = []  # Create ordered list with radii to match the nodes in and intensities
names = []  # Create ordered list with names to match ids
for node in ids:
    r.append(rdf.loc[rdf['ID'] == node]['Radius (mm)'].iloc[0])
    names.append(rdf.loc[rdf['ID'] == node]['Name'].iloc[0])

#########
# Calcs #
#########

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

# Eval for overlap and compute DAT EFFICIENCY ohhhhhhhhhhh baby here we go
def eval(x, y, z):
    new_x, new_y, new_z = x.copy(), y.copy(), z.copy()  # Need new x and y and z in case we have overlap

    ol = overlap(new_x, new_y, new_z)  # Check if we have overlap; returns [overlap: bool, index1: int/None, index2: int/None] where indices are overlaping spheres
    # If we do have overlap, set new random coordinates for one of the spheres and recheck
    while ol[0]:
        new_x[ol[1]] = get_rand_num(lower_bound, upper_bound)
        new_y[ol[1]] = get_rand_num(lower_bound, upper_bound)
        new_z[ol[1]] = get_rand_num(lower_bound, upper_bound)

        ol = overlap(new_x, new_y, new_z)

    # Calculate distances between spheres and invert for efficiency calc later
    inv_dists = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            inv_dists[i,j] = 1/sphere_dist(x, y, z, i, j)

    # print(pd.DataFrame(inv_dists))
    # print(pd.DataFrame(intens))
    efficiency = np.sum(np.multiply(inv_dists, intens))  # Element-wise multiplication of the two matricies so we get efficiency = intens/distance
    # print(efficiency)
    # input()

    # Return new sphere set (adjusted for overlap) + computed efficiency
    return [new_x, new_y, new_z, efficiency]

# Returns [overlap: bool, index1: int/None, index2: int/None] where indices are overlapping spheres
def overlap(x, y, z):  # Implement no-overlap constraint
    for x1, y1, z1 in zip(x, y, z):
        i1 = x.index(x1)  # Get index and radius info for first sphere
        r1 = r[i1]
        for x2, y2, z2 in zip(x, y, z):
            if x1 == x2 or y1 == y2 or z1 == z2:  # Skip same coords
                continue

            i2 = x.index(x2)  # Get index and radius info for second sphere
            r2 = r[i2]

            dist = sphere_dist(x, y, z, i1, i2)  # Compute distance between spheres
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

        ex, ey, ez, e = eval(x, y, z)  # Evaluate sphere set and efficiency

        # Adjust new max_efficiency and let us know we found one
        if max_efficiency < e:
            max_efficiency = e
            max_x_set = ex
            max_y_set = ey
            max_z_set = ez
            print(i, max_efficiency)
        
        # Print out an update every 500
        if i % 500 == 0:
            print(i)

    # Return that biznatch
    return max_efficiency, max_x_set, max_y_set, max_z_set

iterations = 100000
max_efficiency, x, y, z = crude_monte_carlo(iterations)
print(max_efficiency)

# Plot maximally efficient configuration (stolen off the internet lol)
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

df = pd.DataFrame({'x': x, 'y': y, 'z': z})  # Create dataframe to save our endpoints

timestamp = time.strftime('%Y%m%d-%H%M%S')

df.to_csv(f'./results/hy_monte_carlo_efficiency_{iterations}iterations_{max_efficiency}efficiency{timestamp}.csv', index=None)
plt.savefig(f'./results/hy_monte_carlo_efficiency_{iterations}iterations_{max_efficiency}efficiency{timestamp}.png', dpi=300)
plt.show()