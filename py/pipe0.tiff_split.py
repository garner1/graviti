#######################
#Run as:
#python3.7 pipe0.tiff_split.py ~/Work/dataset/tissue2graph/tissues iMS337_20190709_001 ~/Work/dataset/tissue2graph/tissues/tiles 500
######################
import os
import sys
from PIL import Image
import numpy as np
Image.MAX_IMAGE_PIXELS = None #needed to load large tiff files

in_path = sys.argv[1] #the input path
in_filename = sys.argv[2] #the input image without suffix .tif
out_path = sys.argv[3] #the path where you want to store output tiles
step = int(sys.argv[4]) #1024 or 500 or alt

tile_size_x = step
tile_size_y = step

im = np.asarray(Image.open(in_path+'/'+in_filename+'.tif'))
xsize, ysize = im.shape
for i in range(0, xsize, tile_size_x):
    for j in range(0, ysize, tile_size_y):
        # print(i//step, j//step)
        img = Image.fromarray(im[i:i+step,j:j+step])
        img.save(out_path+'/'+in_filename+'_r'+str(i//step)+'_c'+str(j//step)+'.tif')
