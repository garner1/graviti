#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy import ndimage, sparse
from scipy.linalg import eigh, inv, logm, norm
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import glob

import networkx as nx
import seaborn as sns; sns.set()

import umap
import warnings
warnings.filterwarnings('ignore')

def edges_rescaling(edges,scale): # edges are mat.data where mat is a sparse scipy matrix
    edges = np.log10(edges) # log rescale weights because they vary over many decades
    edges -= min(edges) # make them positive 
    edges /= max(edges)*1.0/scale # rescale from 0 to scale
    return edges

N=100000
data = np.load('/home/garner1/Work/pipelines/tissue2graph/npz/X-XY_data.npz',allow_pickle=True)
pos = data['XY'][:N,:]

filename='/home/garner1/Work/pipelines/tissue2graph/npz/mat_XY_10nn.npz'
XY = sparse.load_npz(filename) 
gXY = XY.copy()[:N,:N].tocoo()  #make a copy of the initial data

filename='/home/garner1/Work/pipelines/tissue2graph/npz/mat_X_10nn.npz'
X = sparse.load_npz(filename) 
gX = X.copy()[:N,:N].tocoo()  #make a copy of the initial data

# hada = gX.multiply(gXY)
# hada.data = edges_rescaling(hada.data,1.0) # rescale to a log-scale from 0 to 10

print('geometric graph construction')
G = nx.from_scipy_sparse_matrix(gXY) # if sparse matrix
eset = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight']>0]
weights = [d['weight'] for (u, v, d) in G.edges(data=True) if d['weight']>0]
sns.distplot(weights)
plt.savefig("./d1.png") # save as png
plt.close()
sns.set(style='white', rc={'figure.figsize':(50,50)})
nx.draw_networkx_nodes(G, pos,alpha=0.0) 
nx.draw_networkx_edges(G, pos, edgelist=eset,alpha=1.0, width=weights,edge_color='r',style='solid')
plt.axis('off')
print('saving graph')
plt.savefig("./g1.png") # save as png
plt.close()

print('morphological graph construction')
G = nx.from_scipy_sparse_matrix(gX) # if sparse matrix
eset = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight']>0]
weights = [d['weight'] for (u, v, d) in G.edges(data=True) if d['weight']>0]
sns.distplot(weights)
plt.savefig("./d2.png") # save as png
plt.close()
sns.set(style='white', rc={'figure.figsize':(50,50)})
nx.draw_networkx_nodes(G, pos,alpha=0.0) 
nx.draw_networkx_edges(G, pos, edgelist=eset,alpha=1.0, width=weights,edge_color='r',style='solid')
plt.axis('off')
print('saving graph')
plt.savefig("./g2.png") # save as png
plt.close()
