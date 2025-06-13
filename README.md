# Oocyst prediction

This script performs cell segmentation on 3D microscopy images stored in Imaris (.ims) format using [Cellpose](https://www.cellpose.org/), identifying the most in-focus Z-slice and predicting on that 2D image.

## Installation

```
pip install -r requirements.txt
```

## Usage
```
python predict_oocyst_counts.py inputs.csv output_dir
```

Required Arguments:

input.csv: CSV with column file listing image paths.

output_dir: Directory where segmented .nrrd files will be saved.

Optional Arguments:

--target_spacing: Tuple of XY spacing to select resolution from .ims.

--flow_threshold: Maximum allowed error in Cellpose flows.

--cellprob_threshold: Probability threshold to consider a region a cell.

--tile_norm_blocksize: Block size for image normalization.

## Outputs:

For each .ims file:

filename.nrrd: Original image at highest resolution.

filename_seg.nrrd: Predicted label mask (3D with 2D mask inserted at Z). 

Input CSV is updated with a new column predicted_num_cells.
    
To visualize the segmentations, open filename.nrrd and filename_seg.nrrd using [ITK-SNAP](https://www.itksnap.org/)