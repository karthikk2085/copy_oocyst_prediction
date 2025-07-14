import pathlib
import sys
import numpy as np
import pandas as pd
import SimpleITK as sitk
from sitk_ims_file_io import read, read_metadata
from cellpose import models
import argparse
from functools import partial

"""
Script to predict number of cells from a given list of files in an input csv
file with column name 'file'.

This script performs cell segmentation on 3D microscopy images stored in Imaris (.ims)
format using Cellpose (V3), identifying the most in-focus Z-slice and predicting on that 2D image.
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
    z_slice_in_focus = get_z_slice_index_in_focus(img)
    img_array = sitk.GetArrayFromImage(img[:, :, z_slice_in_focus])

    # Predict on the identified slice
    masks, flows, styles = predictor(img_array)

    # Remove objects touching the image border, they are partial segmentations so not counted.
    image_boundary_arr = np.pad(
        np.zeros([sz - 2 for sz in masks.shape], dtype=np.uint8),
        pad_width=1,
        mode="constant",
        constant_values=1,
    )
    boundary_values_arr = np.multiply(masks, image_boundary_arr)
    masks[np.isin(masks, np.unique(boundary_values_arr[boundary_values_arr != 0]))] = 0
    filtered_mask = sitk.GetImageFromArray(masks.astype(np.uint16))

    filtered_mask.SetOrigin(img.GetOrigin()[:2])
    filtered_mask.SetSpacing(img.GetSpacing()[:2])

    return filtered_mask, z_slice_in_focus


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Segment oocysts using cellpose and count the number of instances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        help="Determines the probability that a detected object is a cell.",
    )
    parser.add_argument(
        "--tile_norm_blocksize",
        default=0,
        help="Determines the size of blocks used for normalizing the image. ",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv_path)

    # Load the model
    model = models.CellposeModel(gpu=True)
    predictor = partial(
        model.eval,
        batch_size=1,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        normalize={"tile_norm_blocksize": args.tile_norm_blocksize},
    )

    predicted_num_cells = []
    for i, file in enumerate(df["file"]):
        try:
            # Obtain the image at the target resolution
            img = read_image(file, target_spacing=args.target_spacing)

            # Predict cells on the obtained image
            label_mask, z_slice_in_focus = predict_cell_segmentation(img, predictor)

            # Read the input images at highest resolution for conversion to nrrd
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

            # Paste the label mask in Z slice.
            full_res_label_mask_3d = sitk.Image(full_res_img.GetSize(), sitk.sitkUInt16)
            full_res_label_mask_3d.CopyInformation(full_res_img)
            full_res_label_mask_3d[:, :, z_slice_in_focus] = full_res_label_mask

            # Save the original image and label mask at the highest resolution for visualization purposes.
            # Avoid overwriting of files with same filenames

            if not (
                pathlib.Path(args.output_dir) / (pathlib.Path(file).stem + ".nrrd")
            ).exists():
                sitk.WriteImage(
                    full_res_img,
                    str(
                        pathlib.Path(args.output_dir)
                        / (pathlib.Path(file).stem + ".nrrd")
                    ),
                )
            else:
                print(
                    str(
                        pathlib.Path(args.output_dir)
                        / (pathlib.Path(file).stem + ".nrrd")
                    )
                    + " already exists. "
                    + "Skipping overwriting the file for "
                    + file
                    + " as they have the same filenames."
                )
                predicted_num_cells.append("")
                continue

            sitk.WriteImage(
                full_res_label_mask_3d,
                str(
                    pathlib.Path(args.output_dir)
                    / (pathlib.Path(file).stem + "_seg.nrrd")
                ),
            )

            # Get no. of labels from the label mask
            stats = sitk.LabelShapeStatisticsImageFilter()
            stats.Execute(label_mask)

            predicted_num_cells.append(len(stats.GetLabels()))
        except Exception as e:
            print(f"Error occurred while processing: {e}", file=sys.stderr)
            predicted_num_cells.append("")
    df["automated oocyst count"] = predicted_num_cells
    df.to_csv(args.input_csv_path, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
