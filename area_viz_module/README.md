# Area visualization

## Input

* list of h5 files with segmented nuclei
* list of tif images matching the h5 files

## Parameters

* (input) directory containing h5 files
* (input) directory containing tif files
* (output) directory containing npz files
* id label of the WSI
* file name of the report file

## How to run the code
```
$ bash pipeline ~/Work/dataset/tissue2graph/tissues/ID2/{h5,tif,npz} ID2
```

## The pipeline
* pipe \#1: collect centroids,areas and intensities for all the nuclei in the WSI
* pipe \#2: stitches the features collected in the previous step to annotate the WSI at the single nucleus scale
* pipe \#3: construct the UMAP graph from the centroids of the nuclei


