#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy import ndimage, sparse
from scipy.linalg import eigh, inv, logm, norm
import sys
import umap
import glob
import warnings
warnings.filterwarnings('ignore')

import networkx as nx
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, zscore, poisson
from sklearn.preprocessing import normalize


filename = sys.argv[1] #'/home/garner1/Work/dataset/tissue2graph/ID57_data_RC-XY-A-I.npz'
mat_XY = sparse.load_npz(sys.argv[2]) #'./ID57_nn20.npz'

data = np.load(filename,allow_pickle=True)
XY = data['XY'] # spatial coordinates of the nuclei
A = data['A'] # area of the nuclei
I = data['I'] # intensity of the nuclei

S = normalize(mat_XY, norm='l1', axis=1) #create the row-stochastic matrix

vec = A
history = vec
nn = 100
for counter in range(nn):
    vec = S.dot(vec)
    history = np.hstack((history,vec)) 

##########################################                       
# Fit a normal distribution to the data:
attribute = np.log2(np.mean(history[:,:],axis=1))
mu, std = norm.fit(attribute) # you could also fit to a lognorma the original data
sns.set(style='white', rc={'figure.figsize':(5,5)})
plt.hist(attribute, bins=100, density=True, alpha=0.6, color='g')
#Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)
plt.savefig("distro.png") # save as png
plt.close()
###########################################

# create empty list for node colors
pos = XY
G = nx.Graph()
G.add_nodes_from(range(len(attribute)))

attribute = np.log2(np.mean(history[:,:],axis=1)) # set the node attribute

# color attribute based on percentiles, deciles or quartiles ...
# node_color = np.interp(attribute, (attribute.min(), attribute.max()), (0, +100))
node_color = pd.qcut(attribute, 10, labels=False)

# draw graph with node attribute color
sns.set(style='white', rc={'figure.figsize':(50,50)})
nx.draw_networkx_nodes(G, pos, alpha=0.5,node_color=node_color, node_size=2,cmap='viridis')

print('saving graph')
plt.axis('off')
plt.savefig("graph-heatmap.png") # save as png
plt.close()

