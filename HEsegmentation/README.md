# Nuclei segmentation of HE-stained images

The python (2.0) script hematoxylin-segmentation.py uses the package histomicsTK to
1. Deconvolve into hematoxylin and eosin components the image
2. Use the hematoxylin component to segment nuclei
3. Generate as output an array Sx2, where S is the number of nuclei

The input is the image file to segment: /path/to/img

The output is an .npz file containg the 'X' array with nuclei centroids: /path/to/img.npz

## How to run
``` parallel "python hematoxylin-segmentation.py {}" ::: ~/Work/dataset/iMS423_20191002_001/*.tif ```

# UMAP graphs creations
The python (3.7) script data2graph.py maps
* input: /path/to/img.npz
into a UMAP graph
* /path/to/img.npz_2graph_nn??.npz