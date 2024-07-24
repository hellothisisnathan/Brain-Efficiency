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
rij = np.zeros((dim,dim))
rij[rij<1.e-6] = 1.e-6
Rij = np.zeros((dim,dim))
Iij = np.zeros((dim,dim)) 

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
rij = (rij + rij.transpose())/2.		
Rij = (Rij + Rij.transpose())/2.
Iij = (Iij + Iij.transpose())/2.
Iij = Iij / 1000


## define the scalar to be minimized
## S = sum of all the edge distances with their intensities
def S(xij):
	return np.dot(Iij.flatten(),(-1./xij).flatten())
	

def constraint(xij):
	return (-Rij.flatten() + xij.flatten())

sol = minimize(S,rij.flatten(),method='SLSQP',bounds=Bounds(0,np.inf,keep_feasible=True),constraints=({'type':'ineq','fun':constraint}),options={'disp':True})

rij_opt = np.reshape(sol.x,(dim,dim))

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

	
fig, ax = plt.subplots()
ax.scatter(Rij.flatten(),rij_opt.flatten(),alpha=0.2)
# ax.set_xlim(-2,8)
# ax.set_ylim(-2,16)
# ax.set_xlabel(r'$r_{ij}$ (true)')
# ax.set_ylabel(r'$r_{ij}$ (optimal solution)')
plt.xlabel(r'$Distance_{ij}$ (minimum)', fontsize=18)
plt.ylabel(r'$Distance_{ij}$ (optimal solution)', fontsize=18)
plt.title('Optimal Model Arrangement vs. True Distance', fontsize=18)
plt.show()

	
fig, ax = plt.subplots()
ax.scatter(df['distance'],df['opt. distance'],alpha=0.2)
# ax.set_xlim(-2,8)
# ax.set_ylim(-2,16)
# ax.set_xlabel(r'$r_{ij}$ (true)')
# ax.set_ylabel(r'$r_{ij}$ (optimal solution)')
plt.xlabel(r'$Distance_{ij}$ (true)', fontsize=18)
plt.ylabel(r'$Distance_{ij}$ (optimal solution)', fontsize=18)
plt.title('Optimal Model Arrangement vs. True Distance', fontsize=18)
plt.show()
fig.savefig('rij_true_vs_OptimalSolution.png')	
	
	
	
fig, ax = plt.subplots()
ax.scatter(df['opt. distance']-df['distance'],df['intensity'],color='k',alpha=0.4)
ax.set_xlabel(r'$\Delta r_{ij}$')
ax.set_ylabel(r'intensity')
plt.show()	