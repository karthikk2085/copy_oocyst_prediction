import pathlib
import sys
import numpy as np
import pandas as pd
import SimpleITK as sitk
from sitk_ims_file_io import read, read_metadata
from cellpose import models
import argparse
from functools import partial
from datetime import datetime

"""
Script to predict number of cells from a given list of files in an input csv
file with column name 'file'. File paths are either absolute or relative to
the location of the input csv file.

This script performs cell segmentation on 3D microscopy images stored in Imaris (.ims)
format using Cellpose (V3), identifying the most in-focus Z-slice and predicting on that 2D image.
"""

# Mean diameter of oocysts in pixels where cellpose performs well.
# Cellpose expects the cell diameter to be about 10 pixels.
DESIRED_DIAMETER_OF_OOCYSTS_IN_PIXELS = 10


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


def file_or_dir(path):
    p = pathlib.Path(path)
    if p.is_file() or p.is_dir():
        return p
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid argument ({path}), not a file or directory."
        )


def create_input_csv(input_csv_path_or_dir):
    """
    Create a csv file in the given directory with a column titled 'file' and entries are all
    the ims files in the directory. File name is timestamped to avoid overwriting.
    Returns the path to the created csv file.
    """
    ims_files = []
    # Create a dataframe with all ims files in the directory
    for f in input_csv_path_or_dir.glob("*"):
        try:
            # Read the first pixel in the first channel of the input file, ensures the file is a
            # valid ims file. Will fail if not an ims file (another file type or a directory
            # returned by glob).
            read(
                f,
                channel_index=0,
                sub_ranges=[range(0, 1), range(0, 1), range(0, 1)],
            )
            ims_files.append(f.name)
        except Exception:
            pass
    df = pd.DataFrame(ims_files, columns=["file"])
    output_path = (
        input_csv_path_or_dir
        / f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_oocyst_counts.csv"
    )
    df.to_csv(output_path, index=False)
    return output_path


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


def predict_cell_segmentation(img, slice_in_focus, predictor):
    """
    Perform cell segmentation on the most in-focus Z slice of a 3D image.

    Parameters:
    - img (SimpleITK.Image): 3D image volume.
    - predictor (Callable): Cell segmentation predictor function
      and returns (masks, flows, styles).

    Returns:
    - SimpleITK.Image: 2D label mask image.
    """

    img_array = sitk.GetArrayFromImage(img[:, :, slice_in_focus])

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

    return filtered_mask


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Segment oocysts using cellpose and count the number of instances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_csv_path_or_dir",
        type=file_or_dir,
        help="Input csv file containing file names with column titled 'file' or a directory with imaris files."
        + "The file paths in the csv are either absolute or paths relative to the csv file location.",
    )
    parser.add_argument(
        "average_physical_diameter_size_of_oocysts",
        type=float,
        help="Average physical diameter size of oocysts in nanometers.",
    )
    parser.add_argument(
        "output_dir",
        type=dir_path,
        help="Output directory to save the label masks.",
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

    # Read the input csv file or create one if a directory is given
    if args.input_csv_path_or_dir.is_file():
        input_csv_path = csv_path(args.input_csv_path_or_dir, required_columns=["file"])
    else:  # this is a directory (argparse ensured this is a dir or file)
        input_csv_path = create_input_csv(args.input_csv_path_or_dir)

    df = pd.read_csv(input_csv_path)

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
    csv_absolute_path = input_csv_path.absolute().parent

    target_spacing = [
        args.average_physical_diameter_size_of_oocysts
        / DESIRED_DIAMETER_OF_OOCYSTS_IN_PIXELS
    ] * 2

    for i in range(len(df)):
        x = df.iloc[i]
        file = x["file"]

        # File names listed in the csv input file are either absolute
        # paths or relative to the csv location.
        if not pathlib.Path(file).is_file():
            file = str((csv_absolute_path / file).resolve())
        try:
            img = read_image(file, target_spacing=target_spacing)

            # Slice in focus should be overriden by csv values if the column values exists,
            # otherwise the computed value for slice in focus within the volume is taken for input.
            if "slice_in_focus" in x and pd.notnull(x["slice_in_focus"]):
                z_slice_in_focus = int(x["slice_in_focus"])
            else:
                # Compute slice in focus
                z_slice_in_focus = get_z_slice_index_in_focus(img)

            # Predict cells on the image at the desired resolution and z slice
            label_mask = predict_cell_segmentation(img, z_slice_in_focus, predictor)

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
                    "Skipping overwriting the file for "
                    + file
                    + " as the filename with "
                    + pathlib.Path(file).stem
                    + " already exists in the output directory."
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
    df.to_csv(input_csv_path, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
