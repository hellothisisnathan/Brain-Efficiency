import pandas as pd
import numpy as np
from itertools import product
from scipy.optimize import minimize
from scipy.optimize import Bounds
import matplotlib.pyplot as plt
font = {
        'family': 'serif',
        'size': 18,
        'style': 'normal',
        'weight': 'medium',
        'fontname': 'Arial'
}


df = pd.read_csv("edge_withDistancesAndRadii.csv")



# fig,ax = plt.subplots()
# ax.scatter(df['distance'],df['intensity']/df['distance'],alpha=0.49)
# ax.set_xlabel('distance',fontdict=font)
# ax.set_ylabel('efficiency',fontdict=font)
# ax.set_xlim(0,)
# ax.set_ylim(10,)
# ax.set_yscale('log')
# plt.show()



## get the list of nodes from the file
nodes = np.unique(df['n1'].tolist()+df['n2'].tolist())

## dimension
dim = len(nodes)

## define the matrix of distances between nodes. It is symmetric
rij = np.zeros((dim,dim))  # Distance matrix
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
Iij = Iij / 1000

## define the scalar to be minimized
## solution = sum of all the edge distances with their intensities
def solution(xij):
	return np.dot(Iij.flatten(),(-1./xij).flatten())  # Intensity / Distance
	
# Minimum distance based on radii + distance guess must be >= 0
def constraint(xij):
	return (-Rij.flatten() + xij.flatten())

sol = minimize(solution,rij.flatten() + np.random.uniform(-0.5, 0.5, dim*dim),constraints=({'type':'ineq','fun':constraint}),options={'disp':True})
exit()

rij_opt = np.reshape(sol.x,(dim,dim))

r = rij[0,1]/rij_opt[0,1]
for i, j in product(range(dim),range(dim)):
	if i>=j: continue
	#print("d[%d,%d]: %1.2f(true) %1.2f(sol.)"%(i,j,rij[i,j], rij_opt[i,j]*r))
	

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

	
	
fig, ax = plt.subplots()
ax.scatter(df['distance'],df['opt. distance'],color='k',alpha=0.4)
#ax.set_xlim(-2,16)
#ax.set_ylim(-2,16)
ax.set_xlabel(r'$r_{ij}$ (true)')
ax.set_ylabel(r'$r_{ij}$ (optimal solution)')
fig.savefig('./minimization troubleshooting/output.png')	
plt.show()
	
	
	
# fig, ax = plt.subplots()
# ax.scatter(df['opt. distance']-df['distance'],df['intensity'],color='k',alpha=0.4)
# ax.set_xlabel(r'$\Delta r_{ij}$')
# ax.set_ylabel(r'intensity')
# plt.show()