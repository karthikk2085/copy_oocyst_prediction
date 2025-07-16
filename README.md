# Oocyst prediction

This repository contains scripts for analyzing oocyst microscopy images. 

For an overview of all parasite development stages in mosquitoes see J C Beier, "Malaria parasite development in mosquitoes", 1998, doi: [10.1146/annurev.ento.43.1.519](https://www.doi.org/10.1146/annurev.ento.43.1.519).


## Installation

```
pip install -r requirements.txt
```

or minconda

```
conda env create -f environment.yml
```

## Script Listing

### predict_oocyst_counts.py

Count the number of live oocysts in a 3D microscopy image where only a single 2D in-focus slice is analyzed. Input is expected to be in Imaris (.ims) format with cell segmentation in the 2D in-focus slice done using [Cellpose](https://www.cellpose.org/). Estimated oocyst counts are added to the input csv file. The original full resolution image and segmentation are written to a user specified output directory in NRRD format, enabling human oversight of the cell count results. Image and segmentation overlay are readily viewable using [ITK-SNAP](https://www.itksnap.org/).

#### Usage:

```
python predict_oocyst_counts.py input.csv output_dir
```

#### Required Arguments:

* input.csv: CSV file with a column titled *file* where each row lists a path to an image file.
* output_dir: Directory to which image and segmentation are written. 

#### Optional Arguments:

* --target_spacing: Tuple of XY spacing to select level in imaris image pyramid with the most similar pixel sizes.
* --flow_threshold: Maximum allowed error in Cellpose flows.
* --cellprob_threshold: Probability threshold to consider a region a cell.
* --tile_norm_blocksize: Block size for image normalization.
