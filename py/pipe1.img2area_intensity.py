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
dapi_file = sys.argv[2] #this file contains the tif images
# npz_file = sys.argv[3] #this is the output file with spatial and morphological descriptors
# method = sys.argv[4] #choose between covd rotational invariant or not: covdRI or covd 
# report = sys.argv[5] #filename of the output report

fov = h5py.File(h5_file, 'r') # load the current fov segmentation
mask = fov['/exported_watershed_masks'][:]
mask_reduced = np.squeeze(mask, axis=2) #to get rid of the third dimension
dapi_fov= cv2.imread(dapi_file,cv2.IMREAD_GRAYSCALE) #the dapi tif file of the current FOV

#Check which position the current FOV occupies within the big scan
row = h5_file.split('_r',1)[1].split('_c')[0]
col = h5_file.split('_r',1)[1].split('_c')[1].split('.')[0]
mask_label, numb_of_nuclei = label(mask_reduced,return_num=True) # label all connected components in the fov, 0 is background
data = []    #list of centroid coordinates for sc in each fov
counter=0
for region in regionprops(mask_label,intensity_image=dapi_fov):
    counter+=1
    tot_intensity = np.sum(region.intensity_image)
    tot_nnz = np.count_nonzero(region.intensity_image)
    print(row,col,counter,tot_intensity,tot_nnz,region.bbox_area)
     
