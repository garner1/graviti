#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
import glob
import warnings
warnings.filterwarnings('ignore')

path = sys.argv[1] # /home/garner1/Work/dataset/tissue2graph/npz
fovsize = sys.argv[2] #1024 or 512
ID = sys.argv[3] #patient ID

filenames = str(path)+'/*.npz'
RCs=[];As=[];Is=[];XYs=[];Ss=[];Es=[];Ps=[] #as lists
counter=0
for filename in glob.glob(filenames):
    counter+=1
    # print(counter,'of',len(glob.glob(filenames)))
    row = int(filename.split('_r',1)[1].split('_c')[0]) - 1 #since rows and cols are 1-based
    col = int(filename.split('_r',1)[1].split('_c')[1].split('.')[0]) - 1 #since rows and cols are 1-based
    data = np.load(filename,allow_pickle=True)
    area = data['areas']
    intensity = data['intensities']
    XY = data['centroids']
    solidity = data['solidities']
    eccentricity = data['eccentricities']
    perimeter = data['perimeters']
    if XY.shape[0]>0:
        #shift the centroids to account for the fov size
        representation = np.asfarray([(row,col,XY[ind,0]+int(fovsize)*col,XY[ind,1]+int(fovsize)*row,
                                       area[ind], intensity[ind],solidity[ind],eccentricity[ind],perimeter[ind])
                                      for ind in range(len(XY)) if area[ind] > 1])
        RCs.append(representation[:,:2]) #only rows and cols
        XYs.append(representation[:,2:4]) #only XY
        As.append(representation[:,4:5])  #only area
        Is.append(representation[:,5:6])  #only intensity
        Ss.append(representation[:,6:7])
        Es.append(representation[:,7:8])
        Ps.append(representation[:,8:9])        
RC = np.vstack(RCs) # the full RC data as array
XY = np.vstack(XYs) # the full XY data as array
A = np.vstack(As) # the full area data as array
I = np.vstack(Is) # the full intensity data as array
S = np.vstack(Ss)
E = np.vstack(Es)
P = np.vstack(Ps)
np.savez('./'+str(ID)+'_'+'data.npz',RC=RC,XY=XY,A=A,I=I,S=S,E=E,P=P)


