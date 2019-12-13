#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import warnings
warnings.filterwarnings('ignore')
import cdlib
from cdlib import algorithms
import matplotlib.pyplot as plt
from cdlib.utils import convert_graph_formats
from cdlib import viz
import networkx as nx
from  scipy import sparse
import numpy as np
from module import *

graph = sys.argv[1] # ~/Work/dataset/tissue2graph/tissues/ID52/pat52_covd_spatial_nn19.npz
data = sys.argv[2] # ~/Work/dataset/tissue2graph/tissues/ID52/pat52_covd_X-XY-RC_data.npz
partition_file = sys.argv[3] #ID52_spatial_partition

# In[ ]:
mat = sparse.load_npz(graph) #mat_XY_nn19.npz
pos = np.load(data,allow_pickle=True)['XY'] #X-XY-RC_data.npz

# In[ ]:
g = nx.from_scipy_sparse_matrix(mat) 
eset = [(u, v) for (u, v, d) in g.edges(data=True)]
weights = [d['weight'] for (u, v, d) in g.edges(data=True)]

# In[ ]:
leiden_coms = algorithms.leiden(g,weights=weights)

# In[ ]:
viz_network_clusters(g, leiden_coms, pos,'./graph_partition',min_cell_numb=1)

# In[ ]:
# viz_network_single_clusters(g, leiden_coms, pos,'./graph_partition',min_cell_numb=1)

import pickle
pkl_file=open(partition_file+'.'+str(len(leiden_coms.communities))+'.pkl','wb')
pickle.dump(leiden_coms.communities,pkl_file)
