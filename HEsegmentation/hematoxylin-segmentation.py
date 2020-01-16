#!/usr/bin/env python
# coding: utf-8

import sys

import histomicstk as htk

import numpy as np
import scipy as sp

import skimage.io
import skimage.measure
import skimage.color

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#Some nice default configuration for plots
plt.rcParams['figure.figsize'] = 20, 20
plt.rcParams['image.cmap'] = 'gray'
titlesize = 24

input_image_file = sys.argv[1] #'/home/garner1/Work/dataset/iMS423_20191002_001/tile_x005_y010.tif'  # Easy1.png

im_input = skimage.io.imread(input_image_file)[:, :, :3]

#Reference file for deconvolution
ref_image_file = ('https://data.kitware.com/api/v1/file/57718cc28d777f1ecd8a883c/download')  

im_reference = skimage.io.imread(ref_image_file)[:, :, :3]

# get mean and stddev of reference image in lab space
mean_ref, std_ref = htk.preprocessing.color_conversion.lab_mean_std(im_reference)

# perform reinhard color normalization, it helps to get more localized feature in hematoxylin image
im_nmzd = htk.preprocessing.color_normalization.reinhard(im_input, mean_ref, std_ref)

# create stain to color map
stainColorMap = {
    'hematoxylin': [0.65, 0.70, 0.29],
    'eosin':       [0.07, 0.99, 0.11],
    'dab':         [0.27, 0.57, 0.78],
    'null':        [0.0, 0.0, 0.0]
}

# specify stains of input image
stain_1 = 'hematoxylin'   # nuclei stain
stain_2 = 'eosin'         # cytoplasm stain
stain_3 = 'null'          # set to null of input contains only two stains

# create stain matrix
W = np.array([stainColorMap[stain_1],
              stainColorMap[stain_2],
              stainColorMap[stain_3]]).T

# perform standard color deconvolution
im_stains = htk.preprocessing.color_deconvolution.color_deconvolution(im_nmzd, W).Stains

# get nuclei/hematoxylin channel
im_nuclei_stain = im_stains[:, :, 0]

# segment foreground
foreground_threshold = 60

im_fgnd_mask = sp.ndimage.morphology.binary_fill_holes(im_nuclei_stain < foreground_threshold)

# run adaptive multi-scale LoG filter
min_radius = 5 #10
max_radius = 10 #15

im_log_max, im_sigma_max = htk.filters.shape.cdog(
    im_nuclei_stain, im_fgnd_mask,
    sigma_min=min_radius * np.sqrt(2),
    sigma_max=max_radius * np.sqrt(2)
)

# detect and segment nuclei using local maximum clustering
local_max_search_radius = 5 #10

im_nuclei_seg_mask, seeds, maxima = htk.segmentation.nuclear.max_clustering(
    im_log_max, im_fgnd_mask, local_max_search_radius)

# filter out small objects
min_nucleus_area = 20 #80

im_nuclei_seg_mask = htk.segmentation.label.area_open(
    im_nuclei_seg_mask, min_nucleus_area).astype(np.int)

# compute nuclei properties
objProps = skimage.measure.regionprops(im_nuclei_seg_mask)

print 'Number of nuclei = ', len(objProps)

# collect the centroid
X = [(objProps[i].centroid[0], objProps[i].centroid[1]) for i in range(len(objProps))]

np.savez(sys.argv[1]+'.npz', X=X)
