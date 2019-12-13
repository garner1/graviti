#!/usr/bin/env python
import numpy as np
from scipy import ndimage, sparse
from scipy.linalg import eigh, inv, logm, norm
import numpy as np
import matplotlib.pyplot as plt
import sys

def covd(ims):
    imd = ims.todense()
    [x,y,I] = [ims.row,ims.col,ims.data]  

    Ix = [];Iy = [];Ixx = [];Iyy = []
    for ind in range(len(I)):
        if x[ind]==np.min(x): #consider the border
            Ix.append(imd[x[ind]+1,y[ind]]-0)
            Ixx.append(imd[x[ind]+1,y[ind]]-2*imd[x[ind],y[ind]]+0)
        elif x[ind]==np.max(x):
            Ix.append(0-imd[x[ind]-1,y[ind]])
            Ixx.append(0-2*imd[x[ind],y[ind]]+imd[x[ind]-1,y[ind]])
        else: 
            Ix.append(imd[x[ind]+1,y[ind]]-imd[x[ind]-1,y[ind]])
            Ixx.append(imd[x[ind]+1,y[ind]]-2*imd[x[ind],y[ind]]+imd[x[ind]-1,y[ind]])
        if y[ind]==np.min(x):
            Iy.append(imd[x[ind],y[ind]+1]-0)
            Iyy.append(imd[x[ind],y[ind]+1]-2*imd[x[ind],y[ind]]+0)
        elif y[ind]==np.max(x):    
            Iy.append(0-imd[x[ind],y[ind]-1])
            Iyy.append(0-2*imd[x[ind],y[ind]]+imd[x[ind],y[ind]-1])    
        else: 
            Iy.append(imd[x[ind],y[ind]+1]-imd[x[ind],y[ind]-1])
            Iyy.append(imd[x[ind],y[ind]+1]-2*imd[x[ind],y[ind]]+imd[x[ind],y[ind]-1])

    # descriptors: x,y, intensity, and x,y,xx,yy derivaties
    descriptor = np.array(list(zip(list(x),list(y),list(I),Ix,Iy,Ixx,Iyy))).T
    C = np.cov(descriptor)
    return C

def airm(a,b): #affine invariant Riemannian metric
    w, v = eigh(a)
    wsqrt = np.sqrt(np.abs(w))
    a1 = inv(v @ np.diag(wsqrt) @ v.T)
    L = norm( logm( a1 @ b @ a1 ) )
    return L

def lerm(a,b): #log euclidean Riemannian metric
    L = norm( logm(a) - logm(b) )
    return L
###################################
filename = sys.argv[1] # 'iMS266_20190426_001.sub3.tif_data_r0_c2.npz'
data = np.load(filename,allow_pickle=True)

covds = []
ids = []
N = len(data['sintensity'])
for ind in range(N):
#     print(str(ind)+' of '+str(N))
    mat = data['sintensity'][ind].tocoo()
    if mat.getnnz() > 0:
        mats = sparse.tril(covd(mat))
        covds.append(mats)
        ids.append(ind)
np.savez(str(sys.argv[1])+'_covds'+'.npz',covds=covds,ids=ids)
