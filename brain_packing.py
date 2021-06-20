import cvxpy as cvx
import numpy as np
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
Iij[20, 15] = 100000
Rij[Rij<1e-6] = 1e-6
Iij = Iij / 1000

c = cvx.Variable(dim*dim)
constr = []
# for i in range(dim):
#     for j in range(i, dim):
#         constr.append(c[i, j] >= Rij[i, j])
for i in range(dim*dim):
	constr.append(c[i] >= Rij.flatten()[i])
prob = cvx.Problem(cvx.Maximize(cvx.norm(cvx.multiply(Iij.flatten(), cvx.inv_pos(c)), 1)), constr)
#prob = cvx.Problem(cvx.Minimize(cvx.max(cvx.max(cvx.abs(cvx.multiply(cvx.inv_pos(cvx.multiply(1./Iij.flatten(), c)), 1/100))))), constr)
prob.solve(method="dccp", solver=cvx.MOSEK, verbose=True)

rij_opt = np.reshape(c.value, (dim, dim))

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

print(cvx.norm(cvx.multiply(Iij.flatten(), cvx.inv_pos(c)), 1).value)
	
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

fig, ax = plt.subplots()
ax.scatter(range(dim*dim), Rij.flatten() - c.value)
plt.show()