#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy import ndimage, sparse
from scipy.linalg import eigh, inv, logm, norm
import sys
import glob
import umap
import warnings
warnings.filterwarnings('ignore')
############################################
# FROM DATA MATRICES X-XY_data.npz TO UMAP GRAPHS
filename = sys.argv[1] # /home/garner1/Work/dataset/tissue2graph/pat52_covd_X-XY-RC_data
method = sys.argv[2] # covd or covdRI
ID = sys.argv[3] #patient ID
graph = sys.argv[4] #graph type

data = np.load(filename+'.npz',allow_pickle=True)
X = data['X'] # log of the descriptor matrix flattened to vector
XY = data['XY'] # spatial coordinates of the nuclei

nn = int(np.log2(X.shape[0])) #scale the nn logarithmically in the numb of nodes to have enough density of edges for clustering
print('UMAP with nn='+str(nn))
if graph=='spatial':
    mat_XY = umap.umap_.fuzzy_simplicial_set(
        XY,
        n_neighbors=nn, 
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
    sparse.save_npz(str(ID)+'_'+str(method)+'_'+str(graph)+'_nn'+str(nn)+'.npz',mat_XY)
if graph=='morphological':
    mat_X = umap.umap_.fuzzy_simplicial_set(
        X,
        n_neighbors=nn, 
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
    sparse.save_npz(str(ID)+'_'+str(method)+'_'+str(graph)+'_nn'+str(nn)+'.npz',mat_X)

