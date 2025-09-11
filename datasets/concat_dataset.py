# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

# Import Path for handling file paths, though it's not directly used in this snippet.
from pathlib import Path

# Import core PyTorch data utilities.
import torch
import torch.utils.data

# Import specific Dataset classes from PyTorch.
# ConcatDataset is used to combine multiple datasets into one.
from torch.utils.data import Dataset, ConcatDataset

# Import the 'build' function from the local 'refexp2seq.py' file.
# This function is responsible for creating datasets for referring expression tasks (like RefCOCO).
from .refexp2seq import build as build_seq_refexp

# Import the 'build' function from the local 'ytvos.py' file.
# This function is responsible for creating the Ref-Youtube-VOS dataset.
from .ytvos import build as build_ytvs

# This import seems redundant as the 'ytvos' module is already imported above.
# It might be a leftover from previous code edits.
from datasets import ytvos


def build(image_set, args):
    """
    This function constructs a single, large dataset by concatenating several smaller ones.
    It combines all RefCOCO variants and the Ref-Youtube-VOS dataset.

    Args:
        image_set (str): The data split to use (e.g., 'train', 'val').
        args: Command-line arguments containing other dataset configurations.

    Returns:
        ConcatDataset: A single dataset object that contains all the specified datasets.
    """
    # Initialize an empty list to hold the individual dataset objects.
    concat_data = []

    # Log that the RefCOCO datasets are being prepared.
    print("preparing coco2seq dataset ....")
    # Define the names of the RefCOCO dataset variants to be loaded.
    coco_names = ["refcoco", "refcoco+", "refcocog"]
    # Loop through each RefCOCO dataset name.
    for name in coco_names:
        # Call the build function for referring expression datasets to create an instance.
        coco_seq = build_seq_refexp(name, image_set, args)
        # Add the created dataset to the list.
        concat_data.append(coco_seq)

    # Log that the Ref-Youtube-VOS dataset is being prepared.
    print("preparing ytvos dataset  .... ")
    # Call the build function for the YTVOS dataset to create an instance.
    ytvos_dataset = build_ytvs(image_set, args)
    # Add the created dataset to the list.
    concat_data.append(ytvos_dataset)

    # Use PyTorch's ConcatDataset to combine all the individual datasets in the list
    # into a single, unified dataset object.
    concat_data = ConcatDataset(concat_data)

    # Return the final concatenated dataset.
    return concat_data
