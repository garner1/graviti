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
# import PyQt5
# PyQt5.QtWidgets.QApplication.setAttribute(PyQt5.QtCore.Qt.AA_EnableHighDpiScaling, True)

from module import *

# In[52]:
K0 = sparse.load_npz(sys.argv[1])  #load the graph ex. mat_XY_10nn.npz
data = np.load(sys.argv[2],allow_pickle=True) #the covd and coordinate data X-XY_data.npz
threshold = float(sys.argv[3]) #threshold for edge removal
mode = sys.argv[4] #if considering full graph or reduced to N
ID = sys.argv[5] #patient ID

#To consider the entire graph:
if mode == 'full':
    K = K0.copy().tocoo()  #make a copy of the initial data
    m = np.ones(K0.shape[0]) #initial masses
    pos = data['XY']

#To consider a smaller graph:
if mode == 'restricted':
    N = 1000
    K = K0.copy()[:N,:N].tocoo()  #make a copy of the initial data
    m = np.ones(K0.shape[0])[:N] #initial masses
    pos = data['XY'][:N,:]

#Initialize omegaIJ/I
omegaIJ = build_omegaij(K.data,K.row,K.col,m)
omegaI = build_omegai(K,m) 
N = np.count_nonzero(omegaI)

#Delete light edges to speedup computation:
idxs_todelete = np.where(np.abs(K.data) < threshold)[0]
print(K.data.shape)
K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col = delete_nodes(K.data,K.row,K.col,omegaIJ.data,omegaIJ.row,omegaIJ.col,idxs_todelete)
print(K.data.shape)

# In[53]:
sns.set(style='white', rc={'figure.figsize':(50,50)})
mat = sparse.coo_matrix((K.data, (K.row, K.col)), shape=(K.row.max()+1, K.col.max()+1)) 
G = nx.from_scipy_sparse_matrix(mat) # if sparse matrix
eset = [(u, v) for (u, v, d) in G.edges(data=True)]
weights = [d['weight'] for (u, v, d) in G.edges(data=True)]
nx.draw_networkx_nodes(G, pos,alpha=0.5)
nx.draw_networkx_edges(G, pos, edgelist=eset,alpha=1.0, width=weights,edge_color='r',style='solid')

plt.axis('off')
plt.savefig('ID'+str(ID)+'_before_RG.png',bbox_inches='tight')
plt.close()

# In[54]:
'''RG flow'''
condition = True
Imax0 = 0
IJmax0 = 0
counter = 0
while condition:
    counter += 1
    condition = np.count_nonzero(omegaI) > N/2 # condition for the loop to stop
    K,omegaI,omegaIJ,Imax,IJmax = renormalization(K,omegaI,omegaIJ,m,threshold) #renormalization step       
    if (counter == 100):
        print('node removal')
        print(counter,Imax,IJmax,np.count_nonzero(omegaI),omegaIJ.data.shape[0])
        counter = 0
        
# In[55]:
fixed_mat = sparse.coo_matrix((K.data, (K.row, K.col)), shape=(K.row.max()+1, K.col.max()+1))
sns.set(style='white', rc={'figure.figsize':(50,50)})
G = nx.from_scipy_sparse_matrix(fixed_mat) 
eset = [(u, v) for (u, v, d) in G.edges(data=True)]
weights = [d['weight'] for (u, v, d) in G.edges(data=True)]
nx.draw_networkx_nodes(G, pos,alpha=0.1)
nx.draw_networkx_edges(G, pos, edgelist=eset,alpha=1.0, width=weights,edge_color='r',style='solid')

plt.axis('off')
plt.savefig('ID'+str(ID)+'_after_RG.png',bbox_inches='tight')
plt.close()





