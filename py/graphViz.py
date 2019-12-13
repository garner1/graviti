#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy import ndimage, sparse
from scipy.linalg import eigh, inv, logm, norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import glob
import networkx as nx
import seaborn as sns; sns.set()
import umap
import warnings
from module import edges_rescaling
warnings.filterwarnings('ignore')

print('morphological graph construction')
mat = sparse.load_npz(sys.argv[1]) # ex: pat52_covd_spatial_nn19.npz
pos = np.load(sys.argv[2],allow_pickle=True)['XY'] #ex: pat52_covd_X-XY-RC_data

G = nx.from_scipy_sparse_matrix(mat) 
eset = [(u, v) for (u, v, d) in G.edges(data=True)]
weights = [d['weight'] for (u, v, d) in G.edges(data=True)]
#nx.write_weighted_edgelist(G, 'graph.weighted.edgelist',delimiter='\t')
sns.set(style='white', rc={'figure.figsize':(50,50)})
nx.draw_networkx_nodes(G, pos, alpha=0.0)
nx.draw_networkx_edges(G, pos, edgelist=eset,alpha=1.0, width=weights,edge_color='r',style='solid')
print('saving graph')
plt.axis('off')
plt.savefig(str(sys.argv[1])+".png") # save as png
plt.close()
