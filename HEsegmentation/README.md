# Nuclei segmentation of HE-stained images

The python (2.0) script uses the package histomicsTK to
1. Deconvolve into hematoxylin and eosin components the image
2. Use the hematoxylin component to segment nuclei
3. Generate as output an array Sx2, where S is the number of nuclei

The input is the image file to segment: /path/to/img

The output is an .npz file containg the 'X' array with nuclei centroids: /path/to/img.npz