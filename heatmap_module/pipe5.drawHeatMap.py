#!/usr/bin/env python
# coding: utf-8
import numpy as np
from scipy import sparse
import sys
import umap
import warnings
import networkx as nx
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, zscore, poisson
from sklearn.preprocessing import normalize
warnings.filterwarnings('ignore')

npyfilename = sys.argv[1] #'/home/garner1/Work/pipelines/graviti/heatmap_module/npy/ID57-area-walkhistory.npy'
npzfilename = sys.argv[2] #'/home/garner1/Work/dataset/tissue2graph/ID57_data_RC-XY-A-I.npz'
feature = sys.argv[3] # one of (area,intensity,perimeter,eccentricity,solidity)
steps = int(sys.argv[4]) # correspond to the size of the nuclei ensemble average: greater the #steps the larger the graph-neighbor-window over which the mean is taken
ID = sys.argv[5] #patient ID
modality = sys.argv[6] #linear or percentiles division of the feature range
scale = sys.argv[7] #linear of logarithmic scale of the attribute values

history = np.load(npyfilename,allow_pickle=True)

if scale == 'linear':
    attribute = np.mean(history[:,:steps],axis=1)
elif scale == 'logarithmic':
    attribute = np.log2(np.mean(history[:,:steps],axis=1))

##########################################                       
# Fit a normal distribution to the data:
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
plt.savefig("./png/"+str(ID)+"_distro-"+str(feature)+"-"+str(scale)+"_scale"+"-nn"+str(steps)+".png") # save as png
plt.close()
###########################################

# create empty list for node colors
G = nx.Graph()
pos = np.load(npzfilename,allow_pickle=True)['XY']
G.add_nodes_from(range(len(attribute)))

# color attribute based on percentiles, deciles or quartiles ...
if modality == 'linear':
    node_color = np.interp(attribute, (attribute.min(), attribute.max()), (0, +10))
elif modality == 'percentiles':
    node_color = pd.qcut(attribute, 10, labels=False)

# draw graph with node attribute color
sns.set(style='white', rc={'figure.figsize':(50,50)})
nx.draw_networkx_nodes(G, pos, alpha=0.5,node_color=node_color, node_size=2,cmap='viridis')

print('saving graph')
plt.axis('off')
plt.savefig("./png/"+str(ID)+"_heatmap-"+str(feature)+"-"+str(scale)+"_scale""-"+str(modality)+"-nn"+str(steps)+".png") # save as png
plt.close()

