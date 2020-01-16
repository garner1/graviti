#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)

library(CRImage)
library("EBImage")

f <- args[1]
img = readImage(f)

#convert to grayscale
imgG=EBImage::channel(img,"gray")
#create a mask for white pixel
whitePixelMask=img[,,1]>0.85 & img[,,2]>0.85 & img[,,3]>0.85
#create binary image
imgB=createBinaryImage(imgG,img,method="otsu",numWindows=4,whitePixelMask=whitePixelMask)

segmentationValues=segmentImage(filename=f,maxShape=800,minShape=40,failureRegion=2000,threshold="otsu",numWindows=4)

writeImage(imgG, "sampleG.jpeg")
writeImage(imgB, "sampleB.jpeg")
writeImage(segmentationValues[[1]], "sampleBSegm1.jpeg")
writeImage(segmentationValues[[2]], "sampleBSegm2.jpeg")