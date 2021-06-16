import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Eval 3d distance between spheres at index i1 and i2
def sphere_dist(x, y, z, i1, i2):
    p1 = np.array([x[i1], y[i1], z[i1]])
    p2 = np.array([x[i2], y[i2], z[i2]])
    squared_dist = np.sum((p1-p2)**2, axis=0)
    dist = np.sqrt(squared_dist)  # Distance between sphere1 and sphere2

    return dist

df = pd.read_csv('../../edge_withDistancesAndRadii.csv')  # Edge list with distances and r1+r2 columns
rdf = pd.read_csv('../../all brain volumes.csv')  # CSV with radius info for nodes
file_name = './results/hy_monte_carlo_efficiency_10000iterations_673.9381787743316efficiency20210615-153555'
xyz = pd.read_csv(file_name + '.csv')

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

xyz['ID'] = ids


df['monte carlo distance'] = 0.0
x, y, z = xyz.x.tolist(), xyz.y.tolist(), xyz.z.tolist()

for row in df.itertuples():
    df.at[row.Index, 'monte carlo distance'] = sphere_dist(x, y, z, xyz.loc[xyz['ID'] == row.n1].index[0], xyz.loc[xyz['ID'] == row.n2].index[0])

print(df)

plt.scatter(df['distance'], df['monte carlo distance'])
plt.title('w/o scaling')
plt.xlabel('true distance')
plt.ylabel('monte carlo distance')
plt.xlim([-1, 10])
plt.ylim([-1, 10])
plt.savefig(file_name + '_unscaled' + '.png', dpi=300)
plt.show()

scaling_factor = df['monte carlo distance'].iloc[0] / df['distance'].iloc[0]
# scaling_factor = df['distance'].iloc[0] / df['monte carlo distance'].iloc[0]

df['scaled monte carlo distance'] = 0.0
for row in df.itertuples():
    df.at[row.Index, 'scaled monte carlo distance'] = row._6 * scaling_factor  # Scale monte carlo distance column; _6 = 'monte carlo distance'

print(df)

plt.scatter(df['distance'], df['scaled monte carlo distance'])
plt.title('w/ scaling')
plt.xlabel('true distance')
plt.ylabel('scaled monte carlo distance')
plt.xlim([-1, 10])
plt.ylim([-1, 10])
plt.savefig(file_name + '_scaled' + '.png', dpi=300)
plt.show()