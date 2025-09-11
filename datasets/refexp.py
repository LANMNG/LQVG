# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""

# Import standard and third-party libraries.
from pathlib import Path
import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

# Import project-specific transformation functions.
import datasets.transforms_image as T


class ModulatedDetection(torchvision.datasets.CocoDetection):
    """
    A custom dataset class that extends torchvision's CocoDetection.
    It's designed for referring expression tasks, where each image is associated with a text caption.
    It also ensures that every item returned has at least one valid object instance after augmentations.
    """

    def __init__(self, img_folder, ann_file, transforms, return_masks):
        """
        Initializes the dataset.
        Args:
            img_folder (str): Path to the folder containing images.
            ann_file (str): Path to the COCO-style annotation JSON file.
            transforms (callable): A function/transform that takes in an image and a target and returns a transformed version.
            return_masks (bool): If True, segmentation masks are returned for each object.
        """
        # Initialize the parent CocoDetection class.
        super(ModulatedDetection, self).__init__(img_folder, ann_file)
        # Store the augmentation transforms.
        self._transforms = transforms
        # Create an instance of a helper class to process COCO annotations.
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset at the given index.
        This method includes a loop to ensure that a valid sample (with at least one object) is returned,
        even if data augmentation crops away all objects.
        """
        instance_check = False
        # Loop until a valid sample with at least one object instance is found.
        while not instance_check:
            # Get the raw image and annotations from the parent class.
            img, target = super(ModulatedDetection, self).__getitem__(idx)
            # Get the unique image ID for the current sample.
            image_id = self.ids[idx]
            # Load the full COCO image metadata.
            coco_img = self.coco.loadImgs(image_id)[0]
            # Extract the referring expression (caption) from the metadata.
            caption = coco_img["caption"]
            # Extract the dataset name if it exists.
            dataset_name = coco_img["dataset_name"] if "dataset_name" in coco_img else None
            # Prepare the initial target dictionary.
            target = {"image_id": image_id, "annotations": target, "caption": caption}
            # Use the 'prepare' helper to convert annotations into tensors (boxes, masks, etc.).
            img, target = self.prepare(img, target)
            # Apply data augmentations if any are defined.
            if self._transforms is not None:
                img, target = self._transforms(img, target)
            # Add the dataset name back to the final target.
            target["dataset_name"] = dataset_name
            # Add any other important metadata from the COCO annotations to the target.
            for extra_key in ["sentence_id", "original_img_id", "original_id", "task_id"]:
                if extra_key in coco_img:
                    target[extra_key] = coco_img[extra_key]

            # Check if any valid bounding boxes remain after augmentations (e.g., random cropping).
            # A sample is valid if it has at least one box.
            target["valid"] = torch.tensor([1]) if len(target["area"]) != 0 else torch.tensor([0])

            # If the sample has at least one valid instance, exit the loop.
            if torch.any(target["valid"] == 1):
                instance_check = True
            else:
                # If augmentations removed all objects, pick a new random sample and try again.
                import random

                idx = random.randint(0, self.__len__() - 1)

        # Add a temporal dimension (T=1) to the image tensor to make it compatible with video models.
        # Final image shape: [1, 3, H, W].
        return img.unsqueeze(0), target


def convert_coco_poly_to_mask(segmentations, height, width):
    """
    Helper function to convert COCO's polygon segmentation format into a tensor of binary masks.
    """
    masks = []
    # Iterate over each object's segmentation data.
    for polygons in segmentations:
        # Convert polygon coordinates to Run-Length Encoding (RLE) format.
        rles = coco_mask.frPyObjects(polygons, height, width)
        # Decode RLE to get a binary mask.
        mask = coco_mask.decode(rles)
        # Ensure the mask has a channel dimension.
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        # Merge masks for multi-part objects into a single mask.
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        # Stack all individual masks into a single tensor.
        masks = torch.stack(masks, dim=0)
    else:
        # If there are no masks, return an empty tensor with the correct shape.
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    """
    A callable class that acts as a transform. It converts raw COCO annotations
    into a clean dictionary of tensors (boxes, labels, masks) that the model can use.
    """

    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        # Get image dimensions.
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        caption = target["caption"] if "caption" in target else None

        # Filter out "crowd" annotations, which are large groups of objects.
        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        # Extract bounding boxes and convert from [x, y, w, h] to [x1, y1, x2, y2] format.
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        # Clamp box coordinates to be within the image boundaries.
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # Extract class labels.
        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        # If requested, convert segmentation polygons to binary masks.
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        # Remove any boxes that have zero width or height after clamping.
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]

        # Assemble the final target dictionary.
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if caption is not None:
            target["caption"] = caption
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id

        # Add other useful metadata for evaluation.
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        target["valid"] = torch.tensor([1])  # Mark as valid since we've processed it.
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return image, target


def make_coco_transforms(image_set, cautious):
    """
    Creates a pipeline of data augmentations for training or validation.
    """
    # Define the standard normalization transform.
    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Define scales for resizing augmentations.
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768]
    final_scales = [296, 328, 360, 392, 416, 448, 480, 512]
    max_size = 800

    # Define the augmentation pipeline for the training set.
    if image_set == "train":
        # Optionally add horizontal flipping.
        horizontal = [] if cautious else [T.RandomHorizontalFlip()]
        return T.Compose(
            horizontal
            + [
                # Randomly select one of two augmentation strategies.
                T.RandomSelect(
                    # Strategy 1: Simple random resizing.
                    T.RandomResize(scales, max_size=max_size),
                    # Strategy 2: A more complex combination of resizing and cropping.
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, 600, respect_boxes=cautious),
                            T.RandomResize(final_scales, max_size=640),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    # Define the augmentation pipeline for the validation set.
    if image_set == "val":
        return T.Compose(
            [
                # Simple resizing and normalization.
                T.RandomResize([360], max_size=640),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")


def build(dataset_file, image_set, args):
    """
    The main factory function to build the referring expression dataset.
    """
    # Get the root path of the COCO dataset.
    root = Path(args.coco_path)
    assert root.exists(), f"provided COCO path {root} does not exist"
    mode = "instances"
    dataset = dataset_file
    # Define the paths to the image folders and annotation files for train/val splits.
    PATHS = {
        "train": (root / "train2014", root / dataset / f"{mode}_{dataset}_train.json"),
        "val": (root / "train2014", root / dataset / f"{mode}_{dataset}_val.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    # Instantiate the ModulatedDetection dataset with the appropriate transforms.
    dataset = ModulatedDetection(
        img_folder,
        ann_file,
        transforms=make_coco_transforms(image_set, False),
        return_masks=args.masks,
    )
    return dataset
