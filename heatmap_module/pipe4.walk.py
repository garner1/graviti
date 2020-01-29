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

filename = sys.argv[1] #'/home/garner1/Work/dataset/tissue2graph/ID57_data_RC-XY-A-I.npz'
mat_XY = sparse.load_npz(sys.argv[2]) #'./ID57_nn20.npz'
steps = sys.argv[4] #number of steps of the random walker 
ID = sys.argv[5] #patient ID

data = np.load(filename,allow_pickle=True)
XY = data['XY'] # spatial coordinates of the nuclei
A = data['A'] # area of the nuclei
I = data['I'] # intensity of the nuclei
P = data['P'] # perimeter
E = data['E'] # eccentricity
S = data['S'] # solidity

SS = normalize(mat_XY, norm='l1', axis=1) #create the row-stochastic matrix

if sys.argv[3] == 'area':
    vec = A
elif sys.argv[3] == 'intensity':
    vec = I
elif sys.argv[3] == 'perimeter':
    vec = P
elif sys.argv[3] == 'eccentricity':
    vec = E
elif sys.argv[3] == 'solidity':
    vec = S

history = vec
nn = int(steps)
for counter in range(nn):
    vec = SS.dot(vec)
    history = np.hstack((history,vec)) 

filename = str(ID)+'-'+str(sys.argv[3])+'-walkhistory'    
np.save('./npy/'+str(filename),history)
