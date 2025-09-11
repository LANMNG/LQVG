# Import necessary PyTorch and Torchvision libraries.
import torch.utils.data
import torchvision

# Import the specific 'build' functions from other dataset files in this package.
# Each 'build' function is responsible for creating a specific dataset instance.
# They are renamed to avoid naming conflicts.
from .rsvg import build as build_rsvg
from .rsvg_mm import build as build_rsvg_mm


def get_coco_api_from_dataset(dataset):
    """
    Helper function to retrieve the COCO API object from a dataset.
    Some datasets might be wrapped in other PyTorch dataset classes like `Subset`.
    This function iteratively unwraps the dataset to find the base COCO object.
    """
    # Loop to handle nested datasets (e.g., a dataset wrapped in multiple `Subset` instances).
    for _ in range(10):
        # This part is commented out but would have been an early exit condition.
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        # If the current dataset object is a `Subset`, get the underlying dataset.
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    # After unwrapping, check if the base dataset is a CocoDetection instance.
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        # If it is, return its `coco` attribute, which is the COCO API object.
        return dataset.coco


def build_dataset(dataset_file: str, image_set: str, args):
    """
    This is a factory function that constructs and returns the correct dataset.
    It acts as a single entry point for creating any dataset supported by the project.

    Args:
        dataset_file (str): The name of the dataset to build (e.g., 'rsvg').
        image_set (str): The split of the dataset to use (e.g., 'train' or 'val').
        args: Command-line arguments containing other dataset configurations.
    """
    # Check the dataset name and call the corresponding build function.
    if dataset_file == 'rsvg':
        # If the dataset is 'rsvg', call the build function imported from `rsvg.py`.
        return build_rsvg(image_set, args)
    if dataset_file == 'rsvg_mm':
        # If the dataset is 'rsvg_mm', call the build function imported from `rsvg_mm.py`.
        return build_rsvg_mm(image_set, args)

    # If the dataset_file name doesn't match any known datasets, raise an error.
    raise ValueError(f'dataset {dataset_file} not supported')
