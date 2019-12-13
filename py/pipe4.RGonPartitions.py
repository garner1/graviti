#!/usr/bin/env python
# coding: utf-8

# In[1]:
from scipy import ndimage, sparse
from scipy.linalg import eigh, inv, logm, norm
import scipy.sparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
from IPython.display import clear_output
import networkx as nx
import seaborn as sns; sns.set()
import pickle
from functools import reduce

from module import *

#Restrict the graph to the community:
K0_XY = sparse.load_npz(sys.argv[1]) #the graph
c_id = int(sys.argv[2]) #the community selected, is an integer
threshold = float(sys.argv[3]) #the edge threshold
data = sys.argv[4] #the spatial information from X-XY-RC_data.npz
partion_file = sys.argv[5] #the file with the list of node partition
ID = sys.argv[6] #patient ID

with open(partion_file, 'rb') as f:
    communities = pickle.load(f)
c = communities[c_id]

print('size of community '+str(c_id)+': '+str(len(c)))
pos = np.load(data,allow_pickle=True)['XY'][c] 

K = K0_XY[c,:]
K = K.tocsc()[:,c].tocoo()
m = np.ones(K.shape[0]) #initial masses

#Initialize omegaIJ/I
omegaIJ = build_omegaij(K.data,K.row,K.col,m)
omegaI = build_omegai(K,m) 
N = np.count_nonzero(omegaI)

#Delete light edges to speedup computation:
idxs_todelete = np.where(np.abs(K.data) < threshold)[0]
print('edges before thresholding: '+str(K.data.shape))
K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col = delete_nodes(K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col,idxs_todelete)
print('edges after thresholding: '+str(K.data.shape))

sns.set(style='white', rc={'figure.figsize':(50,50)})
mat = sparse.coo_matrix((K.data, (K.row, K.col)), shape=(K.row.max()+1, K.col.max()+1)) 
G = nx.from_scipy_sparse_matrix(mat) # if sparse matrix
eset = [(u, v) for (u, v, d) in G.edges(data=True)]
weights = [d['weight'] for (u, v, d) in G.edges(data=True)]
nx.draw_networkx_nodes(G, pos,alpha=0.0)
nx.draw_networkx_edges(G, pos, edgelist=eset,alpha=1.0, width=weights,edge_color='r',style='solid')

plt.axis('off')
plt.title('(nodes,edges): '+str(np.count_nonzero(omegaI))+'; '+str(len(eset)),fontsize=100)
plt.savefig('ID'+str(ID)+'_before_RG_comm'+str(c_id)+'.png',bbox_inches='tight')
plt.close()

'''RG flow'''
condition = True
Imax0 = 0
IJmax0 = 0
counter = 0
while condition:
    counter += 1
    condition = np.count_nonzero(omegaI) > len(c)/2 # condition for the loop to stop
    K,omegaI,omegaIJ,Imax,IJmax = renormalization(K,omegaI,omegaIJ,m,threshold) #renormalization step       
    if (counter == 100):
        print('node removal')
        print(counter,Imax,IJmax,np.count_nonzero(omegaI),omegaIJ.data.shape[0])
        counter = 0

# In[212]:
fixed_mat = sparse.coo_matrix((K.data, (K.row, K.col)), shape=(K.row.max()+1, K.col.max()+1))
sns.set(style='white', rc={'figure.figsize':(50,50)})
G = nx.from_scipy_sparse_matrix(fixed_mat) 
eset = [(u, v) for (u, v, d) in G.edges(data=True)]
weights = [d['weight'] for (u, v, d) in G.edges(data=True)]
nx.draw_networkx_nodes(G, pos,alpha=0.0)
nx.draw_networkx_edges(G, pos, edgelist=eset,alpha=1.0, width=weights,edge_color='r',style='solid')

plt.axis('off')
plt.title('(nodes,edges): '+str(np.count_nonzero(omegaI))+'; '+str(len(eset)),fontsize=100)
plt.savefig('ID'+str(ID)+'_after_RG_comm'+str(c_id)+'.png',bbox_inches='tight')
plt.close()

