from pathlib import Path

# ---------- The path to the dataset of DermaChallenge. This path may change -------------
image_directory = Path("/run/media/leismael/Sosa/School/DermaChallenge")

segmentation_csv_name = "segmentation_dataset.csv"
classification_csv_name = ""

segmentation_csv_path = image_directory / segmentation_csv_name


def get_image_directory() -> Path:
    """Returns the root directory of all images for
    classification and segmentation"""
    if not image_directory.exists():
        raise Exception(
            f"The Path for the dataset folder does not exist. Check it out:\n\tPath: {image_directory}"
        )
    return image_directory


def get_segmentation_csv() -> Path:
    """Returns the Path for the segmentation csv file"""
    if not segmentation_csv_path.exists():
        raise Exception(
            f"The csv file for segmentation was not found. Check the path:\n{segmentation_csv_path}"
        )
    return segmentation_csv_path
