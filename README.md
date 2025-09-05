# Oocyst prediction

This repository contains scripts for analyzing oocyst microscopy images. 

For an overview of all parasite development stages in mosquitoes see J C Beier, "Malaria parasite development in mosquitoes", 1998, doi: [10.1146/annurev.ento.43.1.519](https://www.doi.org/10.1146/annurev.ento.43.1.519).

For an overview of current experimental models for study of malaria see N.V. Simwela and A.P. Waters, "Current status of experimental models for the study of malaria", 2022, doi:[10.1017/S0031182021002134](https://doi.org/10.1017/S0031182021002134) (has a nice figure illustrating the parasite life cycle in mosquito and mammalian host).


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


Count the number of oocysts in a 3D widefield microscopy image after [deconvolution](https://pubmed.ncbi.nlm.nih.gov/16080270/) where only a single 2D in-focus slice is analyzed. Input is expected to be in Imaris (.ims) format with cell segmentation done using [Cellpose](https://www.cellpose.org/).

Estimated oocyst counts are written to a csv file. If the input was a csv file the counts will be added to that file. If the input was a directory the counts will be added to a uniquely named csv file in that directory. The original full resolution image and segmentation are written to a user specified output directory in NRRD format, enabling human oversight of the cell count results. Image and segmentation overlay are readily viewable using [ITK-SNAP](https://www.itksnap.org/).

#### Usage:

```
python predict_oocyst_counts.py input.csv cell_diameter_in_physical_units output_dir
```
or
```
python predict_oocyst_counts.py input_dir cell_diameter_in_physical_units output_dir
```


#### Arguments:

* input.csv: CSV file with a column titled *file* where each row lists a path to an image file and optionally provide column titled *slice_in_focus*  where it represents zero based index of the z slice which should be used instead of the automatically identified slice. If the column does not exist or the entry is empty, the program automatically identifies this z slice.
* input_dir: instead of a csv file, provide a directory containing imaris files.
* cell_diameter_in_physical_units: Average cell diameter in physical units (micrometers).
* output_dir: Directory to which image and segmentation are written. 

#### Optional, cellpose, arguments:

* --flow_threshold: Maximum allowed error in Cellpose flows.
* --cellprob_threshold: Probability threshold to consider a region a cell.
* --tile_norm_blocksize: Block size for image normalization.
* --manual_segmentation: Option to manually correct any false positives/negatives missed from predictions