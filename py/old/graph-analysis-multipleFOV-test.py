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

# FROM POSITIONS AND DESCRIPTORS TO DATA MATRICES FOR UMAP
# XYs=[];Xs=[]
# path = "/home/garner1/Work/dataset/tissue2graph/h5_post_watershed/iMS266_20190426_001.sub*_Segmented_mask.h5_data_r*_c*.npz"
# for filename in glob.glob(path):
#     data = np.load(filename,allow_pickle=True)
#     covds = data['descriptors']
#     row=data['row'];col=data['col']
#     XY = data['centroids']
#     print(XY.shape,covds.shape,filename)
#     if XY.shape[0]>0:
#         representation = np.asfarray([np.hstack(([XY[ind,0]+1024*col[0],XY[ind,1]+1024*row[0]], logm(covds[ind]).flatten())) for ind in range(len(covds)) if np.count_nonzero(covds[ind]) > 0 ])
#         XYs.append(representation[:,:2])
#         Xs.append(representation[:,2:])
# X = np.vstack(Xs) # the full X data 
# XY = np.vstack(XYs) # the full XY data
# print(X.shape,XY.shape)
# np.savez('/home/garner1/Work/pipelines/tissue2graph/npz/X-XY_data.npz',X=X,XY=XY)

# FROM DATA MATRICES X-XY_data.npz TO UMAP GRAPHS
data = np.load('/home/garner1/Work/pipelines/tissue2graph/npz/X-XY_data.npz',allow_pickle=True)
X = data['X']
XY = data['XY']
print('umap')
mat_XY = umap.umap_.fuzzy_simplicial_set(
    XY,
    n_neighbors=10, 
    random_state=np.random.RandomState(seed=42),
    metric='l2',
    metric_kwds={},
    knn_indices=None,
    knn_dists=None,
    angular=False,
    set_op_mix_ratio=1.0,
    local_connectivity=2.0,
    verbose=False
    )
mat_X = umap.umap_.fuzzy_simplicial_set(
    X,
    n_neighbors=10, 
    random_state=np.random.RandomState(seed=42),
    metric='l2',  
    metric_kwds={},
    knn_indices=None,
    knn_dists=None,
    angular=False,
    set_op_mix_ratio=1.0,
    local_connectivity=2.0,
    verbose=False
    )
sparse.save_npz('../npz/mat_X_10nn.npz',mat_X)
sparse.save_npz('../npz/mat_XY_10nn.npz',mat_XY)

# hada = mat_X.multiply(mat_XY)
# hada.data = edges_rescaling(hada.data,1.0) # rescale to a log-scale from 0 to 10

# print('morphological graph construction')
# G = nx.from_scipy_sparse_matrix(hada) # if sparse matrix
# eset = [(u, v) for (u, v, d) in G.edges(data=True)]
# weights = [d['weight'] for (u, v, d) in G.edges(data=True)]
# pos = XY
# sns.set(style='white', rc={'figure.figsize':(50,50)})
# nx.draw_networkx_nodes(G, pos,alpha=0.0) 
# ind=0
# for edge in eset:
#     print(ind)
#     edgepos={edge[0]: (pos[edge[0],0],pos[edge[0],1]), edge[1]: (pos[edge[1],0],pos[edge[1],1])}
#     G.add_nodes_from(edgepos.keys())
#     for n, p in edgepos.items():
#         G.nodes[n]['pos'] = p
#     nx.draw_networkx_edges(G, pos,alpha=weights[ind],width=1.0,edge_color='r',style='solid')
#     ind+=1
# print('saving graph')
# plt.axis('off')
# plt.savefig("./hada.png") # save as png

# print('geometric graph construction')
# G = nx.from_scipy_sparse_matrix(mat_XY) # if sparse matrix
# eset = [(u, v) for (u, v, d) in G.edges(data=True)]
# weights = [d['weight'] for (u, v, d) in G.edges(data=True)]
# pos = XY
# sns.set(style='white', rc={'figure.figsize':(50,50)})
# nx.draw_networkx_nodes(G, pos,alpha=0.0) 
# ind=0
# for edge in eset:
#     print(ind)
#     edgepos={edge[0]: (pos[edge[0],0],pos[edge[0],1]), edge[1]: (pos[edge[1],0],pos[edge[1],1])}
#     G.add_nodes_from(edgepos.keys())
#     for n, p in edgepos.items():
#         G.nodes[n]['pos'] = p
#     nx.draw_networkx_edges(G, pos,alpha=weights[ind],width=1.0,edge_color='r',style='solid')
#     ind+=1
# print('saving graph')
# plt.axis('off')
# plt.savefig("./geometric.png") # save as png

