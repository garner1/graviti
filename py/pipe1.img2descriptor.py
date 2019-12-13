#!/usr/bin/env python
# coding: utf-8
###########################
import h5py
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix,lil_matrix,coo_matrix
import cv2
import os
import sys
import csv
import warnings
from module import *
warnings.filterwarnings('ignore')
##################################
h5_file = sys.argv[1]   #this file contains the segmented nuclei
datadir = os.path.dirname(os.path.realpath(h5_file))
dapi_file = sys.argv[2] #this file contains the tif images
npz_file = sys.argv[3] #this is the output file with spatial and morphological descriptors
method = sys.argv[4] #choose between covd rotational invariant or not: covdRI or covd 
report = sys.argv[5] #filename of the output report

#Get all the FOVs and arrange them in the position
h5_files = []  #this will contains all fov h5 filenames
print(datadir)
for file in os.listdir(datadir):
    if file.endswith(".h5"):
        h5_files.append(os.path.join(datadir, file))

number_fovs=np.arange(1,len(h5_files)+1)        #these are the labels of the fov; to get the range we have to add one
fov_matrix=np.array(number_fovs).reshape(23,25) #reshape labels according to the patched 2D FOV structure

fov = h5py.File(h5_file, 'r') # load the current fov segmentation
mask = fov['/exported_watershed_masks'][:]
mask_reduced = np.squeeze(mask, axis=2) #to get rid of the third dimension
dapi_fov= cv2.imread(dapi_file,cv2.IMREAD_GRAYSCALE) #the dapi tif file of the current FOV

#Check which position the current FOV occupies within the big scan
[other,value]=dapi_file.split('sub')
[value,other]=value.split('.')
(row,col)=np.where(fov_matrix==int(value))
mask_label, numb_of_nuclei = label(mask_reduced,return_num=True) # label all connected components in the fov, 0 is background
centroids = []    #list of centroid coordinates for sc in each fov
descriptors = []  #list of descriptors for sc in each fov
counter=0
for region in regionprops(mask_label,intensity_image=dapi_fov):
    counter+=1
    if np.count_nonzero(region.intensity_image) <= 1:        #at least 1 cell
        print('There is only one nonzero pixels in row='+str(row)+' col='+str(col)+' and region='+str(counter))
    else:
        centroids.append(region.centroid)
        if method == 'covd':
            descriptors.append(covd(region.intensity_image))
        if method == 'covdRI':
            descriptors.append(covd_ri(region.intensity_image))
#save covd to file
from datetime import datetime
# Returns a datetime object containing the local date and time
dateTimeObj = datetime.now()

if numb_of_nuclei > 0:
    np.savez(str(npz_file)+'_'+str(method)+'_r'+str(row[0])+'_c'+str(col[0])+'.npz',row=row,col=col,centroids=centroids,descriptors=descriptors)
else:
    print('There are no nuclei in row='+str(row)+' and col='+str(col)+' in file: '+str(h5_file))
with open(str(report), 'a+', newline='') as myfile:
     wr = csv.writer(myfile)
     wr.writerow([dateTimeObj,'row='+str(row),'col='+str(col),'nuclei='+str(numb_of_nuclei),'#descriptors='+str(len(descriptors))])
     
