import cvxpy as cvx
import numpy as np
import random
import matplotlib.pyplot as plt
import dccp
import pandas as pd
from itertools import product

np.random.seed(0)

df = pd.read_csv('../edge_withDistancesAndRadii.csv')
nodes = np.unique(df['n1'].tolist() + df['n2'].tolist())
dim = len(nodes)
r = np.array([3, 3, 3, 3, 3])

## define the matrix of distances between nodes. It is symmetric
rij = np.zeros((dim,dim))  # Distance matrix
rij[rij<1.e-6] = 1.e-6
Rij = np.zeros((dim,dim))  # R1+R2 combined radii (min distance) matrix
Iij = np.zeros((dim,dim))  # Intensity matrix

for i1 in range(dim):
	for i2 in range(dim):
		if i1 == i2: continue
		tempdf = df[(df['n1']==nodes[i1])&(df['n2']==nodes[i2])]
		if tempdf.index.size > 1:
			print("two edges for (%s,%s). only one will be stored. drop the duplicate and run again"%(nodes[i1],nodes[i2]))
			#raise(Exception("please drup the duplicate and run again"))
		elif tempdf.index.size == 1:
			rij[i1,i2] = tempdf['distance']
			Rij[i1,i2] = tempdf['R1+R2']
			Iij[i1,i2] = tempdf['intensity']

## symmetrize the tensors
rij = (rij + rij.transpose())/2.  # Distance matrix		
Rij = (Rij + Rij.transpose())/2.  # R1+R2 combined radii (min distance) matrix
Iij = (Iij + Iij.transpose())/2.  # Intensity matrix
Iij[Iij<1] = 1
Rij[Rij<1e-6] = 1e-6
Iij = Iij / 1000

################################################################################
################################################################################
################################################################################
top_config = []
top_val = 0
for i in range(5):
    dist = []
    if i % 100 == 1:
        print("i = ", i, top_val, max(top_config))
    for j in range(dim*dim):
        dist.append(Rij.flatten()[j])
        #dist.append(random.uniform(Rij.flatten()[j], 20))
    val = np.dot(Iij.flatten(),  1./np.array(dist))
    if val > top_val:
        top_val = val
        top_config = dist
################################################################################
################################################################################
################################################################################
rij_opt = np.reshape(np.array(top_config), (dim, dim))
print(rij < Rij)
input()

r = rij[0,1]/rij_opt[0,1]

for i, j in product(range(dim),range(dim)):
	if i>=j: continue
	print("d[%d,%d]: %1.2f(true) %1.2f(sol.)"%(i,j,rij[i,j], rij_opt[i,j]*r))
	

df['opt. distance'] = df.index.size*[0]
for i1 in range(dim):
	for i2 in range(dim):
		if i1 == i2: continue
		tempdf = df[(df['n1']==nodes[i1])&(df['n2']==nodes[i2])]
		if tempdf.index.size > 1:
			print("two edges for (%s,%s). only one will be stored. drop the duplicate and run again"%(nodes[i1],nodes[i2]))
			#raise(Exception("please drup the duplicate and run again"))
		elif tempdf.index.size == 1:
			df['opt. distance'][(df['n1']==nodes[i1])&(df['n2']==nodes[i2])] = r*rij_opt[i1,i2]

print("top val", top_val)

fig, ax = plt.subplots()
ax.scatter(df['distance'],df['opt. distance'],alpha=0.2)
ax.set_xlim(-2,8)
ax.set_ylim(-2,16)
# ax.set_xlabel(r'$r_{ij}$ (true)')
# ax.set_ylabel(r'$r_{ij}$ (optimal solution)')
plt.xlabel(r'$Distance_{ij}$ (true)', fontsize=18)
plt.ylabel(r'$Distance_{ij}$ (optimal solution)', fontsize=18)
plt.title('Optimal Model Arrangement vs. True Distance', fontsize=18)
plt.show()
fig.savefig('rij_true_vs_OptimalSolution_MOSEK.png')	
	
	
	
fig, ax = plt.subplots()
ax.scatter(df['opt. distance']-df['distance'],df['intensity'],color='k',alpha=0.4)
ax.set_xlabel(r'$\Delta r_{ij}$')
ax.set_ylabel(r'intensity')
plt.show()
print(top_config)

fig, ax = plt.subplots()
ax.set_xlim(0,6)
ax.set_ylim(0,6)
ax.scatter(rij.flatten(), rij_opt.flatten(), alpha=0.2)
plt.show()

fig, ax = plt.subplots()
ax.scatter(range(dim*dim), rij.flatten() - rij_opt.flatten(), color='k', alpha=0.4)
plt.show()

print(np.dot(rij.flatten(), (Iij*1000).flatten()))