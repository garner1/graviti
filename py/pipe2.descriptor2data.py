#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy import ndimage, sparse
from scipy.linalg import eigh, inv, logm, norm
import sys
import glob
import warnings
warnings.filterwarnings('ignore')

path = sys.argv[1] # /home/garner1/Work/dataset/tissue2graph/npz
method = sys.argv[2] # covd or covdRI
fovsize = sys.argv[3] #1024 or 512
ID = sys.argv[4] #patient ID

filenames = str(path)+'/*'+str(method)+'*.npz'
RCs=[];XYs=[];Xs=[] #as lists
counter=0
for filename in glob.glob(filenames):
    counter+=1
    print(counter,'of',len(glob.glob(filenames)))
    data = np.load(filename,allow_pickle=True)
    covds = data['descriptors']
    row = int(filename.split('_r',1)[1].split('_c')[0]) - 1 #since rows and cols are 1-based
    col = int(filename.split('_r',1)[1].split('_c')[1].split('.')[0]) - 1 #since rows and cols are 1-based
    XY = data['centroids']
    if XY.shape[0]>0:
        #shift the centroids to account for the fov size
        representation = np.asfarray(
            [np.hstack(([row,col,XY[ind,0]+int(fovsize)*col,XY[ind,1]+int(fovsize)*row], logm(covds[ind]).flatten())) for ind in range(len(covds)) if np.count_nonzero(covds[ind]) > 0 ]
        )
        RCs.append(representation[:,:2]) #only rows and cols
        XYs.append(representation[:,2:4]) #only XY
        Xs.append(representation[:,4:])  #only X
X = np.vstack(Xs)   # the full X data as array
XY = np.vstack(XYs) # the full XY data as array
RC = np.vstack(RCs) # the full RC data as array
np.savez('/home/garner1/Work/dataset/tissue2graph/'+str(ID)+'_'+str(method)+'dataX-XY-RC.npz',X=X,XY=XY,RC=RC)


