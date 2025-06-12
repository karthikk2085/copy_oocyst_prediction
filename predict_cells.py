import pathlib
import sys
import numpy as np
import pandas as pd
import SimpleITK as sitk
from sitk_ims_file_io import read, read_metadata
from cellpose import models
from skimage.segmentation import clear_border
import argparse
from functools import partial

"""
Script to predict number of cells from a given list of files in an input csv
file with column name 'file'.

This script reads metadata from a tab-separated CSV file containing acquisition
details (e.g., tile positions, channels, spacing, filenames), constructs 3D tile
volumes per channel for each (X, Y) position in a well, and registers overlapping
tiles into a single montage. The script saves the generated montage in the user
given output directory. It also saves the tile volumes and transforms in the
output directory using the format {well_name}_{row}_{col}.nrrd/.tfm, where
well_name is the well identifier and row and col indicate the tile's grid position.
These files, generated during montage registration, are saved if the optional
--save_registration_info flag is provided.

"""


def csv_path(path, required_columns={}):
    """
    Define the csv_path type for use with argparse. Checks
    that the given path string is a path to a csv file and that the
    header of the csv file contains the required columns.
    """
    p = pathlib.Path(path)
    required_columns = set(required_columns)
    if p.is_file():
        try:  # only read the csv header
            expected_columns_exist = required_columns.issubset(
                set(pd.read_csv(path, nrows=0).columns.tolist())
            )
            if expected_columns_exist:
                return p
            else:
                raise argparse.ArgumentTypeError(
                    f"Invalid argument ({path}), does not contain all expected columns."
                )
        except UnicodeDecodeError:
            raise argparse.ArgumentTypeError(
                f"Invalid argument ({path}), not a csv file."
            )
    else:
        raise argparse.ArgumentTypeError(f"Invalid argument ({path}), not a file.")


def dir_path(path):
    p = pathlib.Path(path)
    if p.is_dir():
        return p
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid argument ({path}), not a directory path or directory does not exist."
        )


def get_z_slice_index_in_focus(image_3d):
    """
    Get the index to the z slice that has the maximal standard deviation, which we assume
    corresponds to having the best focus (least blurred).
    """
    stats_filter = sitk.StatisticsImageFilter()
    max_std = -1.0  # initialize with value lower than the minimal possible
    max_std_index = 0
    for slc_index in range(image_3d.GetDepth()):
        stats_filter.Execute(image_3d[:, :, slc_index])
        current_std = stats_filter.GetSigma()
        if current_std > max_std:
            max_std = current_std
            max_std_index = slc_index
    return max_std_index


def find_nearest_spacing_index(spacing_list, target_spacing):
    """
    Find the index of the spacing in spacing_list that is closest to the target_spacing.

    Parameters:
    - spacing_list (List[Tuple[float, float, ...]]): List of spacing tuples.
    - target_spacing (Tuple[float, float]): Target spacing to compare against.

    Returns:
    - int: Index of the closest spacing in the list.
    """
    spacing_array = np.array([spacing[0:2] for spacing in spacing_list])
    target_array = np.array(target_spacing)
    distances = np.linalg.norm(spacing_array - target_array, axis=1)

    return int(np.argmin(distances))


def read_image(file, target_spacing):
    """
    Read an image from file at the resolution closest to the target spacing.

    Parameters:
    - file (str): Path to the image file.
    - target_spacing (Tuple[float, float]): Desired spacing to match.

    Returns:
    - SimpleITK.Image: The image read at the closest resolution.
    """
    metadata = read_metadata(file)

    img = read(
        file,
        resolution_index=find_nearest_spacing_index(
            metadata["spacings"], target_spacing
        ),
    )

    return img


def predict_cell_segmentation(img, predictor):
    """
    Perform cell segmentation on the most in-focus Z slice of a 3D image.

    Parameters:
    - img (SimpleITK.Image): 3D image volume.
    - predictor (Callable): Cell segmentation predictor function
      and returns (masks, flows, styles).

    Returns:
    - SimpleITK.Image: 2D label mask image.
    """

    # Identify z slice
    img_array = sitk.GetArrayFromImage(img[:, :, get_z_slice_index_in_focus(img)])

    # Predict on the identified slice
    masks, flows, styles = predictor(img_array)

    # Post process the predicted mask
    filtered_mask = sitk.GetImageFromArray(clear_border(masks).astype(np.uint16))

    # filtered_mask.SetOrigin(img.GetOrigin()[:2])

    return filtered_mask


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Predict cells on imaris file using cellpose."
    )
    parser.add_argument(
        "input_csv_path",
        type=lambda x: csv_path(
            x,
            required_columns=["file"],
        ),
        help="Input csv file containing file names with column 'file'",
    )
    parser.add_argument(
        "output_dir",
        type=dir_path,
        help="Output directory to save the label masks.",
    )
    parser.add_argument(
        "--target_spacing",
        type=tuple,
        default=(3.63, 3.63),
        help="Target spacing to obtain the image resolution from the imaris file. ",
    )
    parser.add_argument(
        "--flow_threshold",
        default=0.4,
        help="Maximum allowed error of the flows for each mask. ",
    )
    parser.add_argument(
        "--cellprob_threshold",
        default=0.0,
        help="Determines the probability that a detected object is a cell. The deault is 0.0",
    )
    parser.add_argument(
        "--tile_norm_blocksize",
        default=0,
        help="Determines the size of blocks used for normalizing the image. ",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv_path)

    model = models.CellposeModel(gpu=True)
    predictor = partial(
        model.eval,
        batch_size=1,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        normalize={"tile_norm_blocksize": args.tile_norm_blocksize},
    )

    df["predicted_num_cells"] = [None] * len(df)
    for i, file in enumerate(df["file"]):

        # Obtain the image at the target resolution
        img = read_image(file, target_spacing=args.target_spacing)

        # Predict cells on the obtained image
        label_mask = predict_cell_segmentation(img, predictor)

        full_res_img = read(file)

        # Resample the label mask to original image size in X and Y
        full_res_label_mask = sitk.Resample(
            label_mask,
            full_res_img.GetSize()[:2],
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            full_res_img.GetOrigin()[:2],
            full_res_img.GetSpacing()[:2],
            label_mask.GetDirection(),
            0,
            sitk.sitkUInt16,
        )
        # full_res_label_mask.SetOrigin(full_res_img.GetOrigin()[:2])

        # Paste the label mask in Z slice.
        full_res_label_mask_3d = sitk.Image(full_res_img.GetSize(), sitk.sitkUInt16)
        full_res_label_mask_3d.CopyInformation(full_res_img)
        full_res_label_mask_3d[:, :, get_z_slice_index_in_focus(full_res_img)] = (
            full_res_label_mask
        )

        # Save the input images at highest resolution and the label masks for visualization
        sitk.WriteImage(
            full_res_img,
            str(pathlib.Path(args.output_dir) / (pathlib.Path(file).stem + ".nrrd")),
        )

        sitk.WriteImage(
            full_res_label_mask_3d,
            str(
                pathlib.Path(args.output_dir) / (pathlib.Path(file).stem + "_seg.nrrd")
            ),
        )

        # Get no. of labels from the label mask
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(label_mask)

        df.loc[i, "predicted_num_cells"] = len(stats.GetLabels())

    df.to_csv(args.input_csv_path, index=False)


if __name__ == "__main__":
    sys.exit(main())
