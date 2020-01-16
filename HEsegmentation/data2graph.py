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
filename = sys.argv[1] # path/to/*.tif.npz file

data = np.load(filename,allow_pickle=True)
XY = data['X'] # spatial coordinates of the nuclei

nn = int(np.log2(XY.shape[0])) #scale the nn logarithmically in the numb of nodes to have enough density of edges for clustering
print('UMAP with nn='+str(nn))

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
sparse.save_npz(filename+'_2graph_nn'+str(nn)+'.npz',mat_XY)

