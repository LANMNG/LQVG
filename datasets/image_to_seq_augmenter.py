# ------------------------------------------------------------------------
# This code is modified from previous works: SeqFormer and STEm-Seg.
# It's designed to apply data augmentation to images and their corresponding
# segmentation masks and bounding boxes, often to simulate video sequences from single images.
# ------------------------------------------------------------------------


# Import the core imgaug library for data augmentation.
import imgaug
import imgaug.augmenters as iaa

# Import numpy for numerical operations, especially on image arrays.
import numpy as np

# Import datetime to generate time-based seeds for randomness.
from datetime import datetime

# Import specific classes from imgaug for handling segmentation maps and bounding boxes.
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class ImageToSeqAugmenter(object):
    """
    A class that defines a pipeline of image augmentations.
    It can apply color, geometric (affine, perspective), and motion blur transformations.
    It's specifically designed to handle images along with their segmentation masks and bounding boxes,
    ensuring all are transformed consistently.
    """

    def __init__(
        self,
        perspective=True,
        affine=True,
        motion_blur=True,
        brightness_range=(-50, 50),
        hue_saturation_range=(-15, 15),
        perspective_magnitude=0.12,
        scale_range=1.0,
        translate_range={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
        rotation_range=(-20, 20),
        motion_blur_kernel_sizes=(7, 9),
        motion_blur_prob=0.5,
    ):
        """
        Initializes the augmentation pipeline with various configurable options.
        """

        # Define a basic color augmentation pipeline.
        # It applies EITHER brightness/contrast changes OR hue/saturation changes.
        self.basic_augmenter = iaa.SomeOf(
            (1, None),
            [
                iaa.Add(brightness_range),  # Adjust brightness.
                iaa.AddToHueAndSaturation(hue_saturation_range),  # Adjust color hue and saturation.
            ],
        )

        # Create a list to hold geometric transformations.
        transforms = []
        if perspective:
            # Add a perspective transformation to simulate camera angle changes.
            transforms.append(iaa.PerspectiveTransform(perspective_magnitude))
        if affine:
            # Add affine transformations: scaling, translation, and rotation.
            transforms.append(
                iaa.Affine(
                    scale=scale_range,
                    translate_percent=translate_range,
                    rotate=rotation_range,
                    order=1,  # Use linear interpolation.
                    backend="auto",
                )
            )  # Automatically choose backend (e.g., OpenCV).

        # Combine the geometric transforms into a sequence.
        transforms = iaa.Sequential(transforms)
        transforms = [transforms]  # Wrap in a list to append more augmenters.

        if motion_blur:
            # Define a motion blur augmentation that is applied with a certain probability.
            blur = iaa.Sometimes(
                motion_blur_prob,
                iaa.OneOf(
                    [
                        # Choose one kernel size for the motion blur.
                        iaa.MotionBlur(ksize)
                        for ksize in motion_blur_kernel_sizes
                    ]
                ),
            )
            transforms.append(blur)

        # Combine all transformations (geometric + motion blur) into the final sequence.
        # This is named 'frame_shift_augmenter' because it simulates frame-to-frame changes in a video.
        self.frame_shift_augmenter = iaa.Sequential(transforms)

    @staticmethod
    def condense_masks(instance_masks):
        """
        Static method to convert a list of binary instance masks into a single integer-labeled segmentation map.
        imgaug requires this format to augment multiple masks simultaneously.
        Example: [[0,1,1], [1,0,0]] -> [2, 1, 1] where 1 is the first mask, 2 is the second.
        """
        # Create an empty mask with the same shape as the first instance mask.
        condensed_mask = np.zeros_like(instance_masks[0], dtype=np.int8)
        # Iterate through each binary mask, assigning a unique integer ID (starting from 1).
        for instance_id, mask in enumerate(instance_masks, 1):
            # Where the binary mask is true, set the corresponding pixel in the condensed mask to the instance ID.
            condensed_mask = np.where(mask, instance_id, condensed_mask)

        return condensed_mask

    @staticmethod
    def expand_masks(condensed_mask, num_instances):
        """
        Static method to perform the reverse of condense_masks.
        It converts a single integer-labeled segmentation map back into a list of binary masks.
        """
        # Create a list of binary masks by checking for each instance ID.
        return [(condensed_mask == instance_id).astype(np.uint8) for instance_id in range(1, num_instances + 1)]

    def __call__(self, image, masks=None, boxes=None):
        """
        Applies the defined augmentation pipeline to an image and its optional masks/boxes.
        """
        # Create a deterministic version of the geometric augmenter.
        # This ensures the exact same geometric transformation is applied to the image, masks, and any other spatial data.
        det_augmenter = self.frame_shift_augmenter.to_deterministic()

        # If masks are provided, augment them along with the image.
        if masks is not None:
            masks_np, is_binary_mask = [], []
            boxs_np = []

            # Prepare masks for augmentation.
            for mask in masks:
                if isinstance(mask, np.ndarray):
                    masks_np.append(mask.astype(np.bool_))
                    is_binary_mask.append(False)
                else:
                    raise ValueError("Invalid mask type: {}".format(type(mask)))

            num_instances = len(masks_np)
            # Condense the list of binary masks into a single integer map and wrap it for imgaug.
            masks_np = SegmentationMapsOnImage(self.condense_masks(masks_np), shape=image.shape[:2])

            # Use a time-based seed to ensure the next two augmentations are identical.
            seed = int(datetime.now().strftime("%M%S%f")[-8:])
            imgaug.seed(seed)

            # Augment the image and the condensed masks.
            # Note: Color augmentation (`basic_augmenter`) is applied ONLY to the image.
            # Geometric augmentation (`det_augmenter`) is applied to both.
            aug_image, aug_masks = det_augmenter(image=self.basic_augmenter(image=image), segmentation_maps=masks_np)

            # Reset the seed to apply the same geometric transform again.
            imgaug.seed(seed)
            # Create a mask of valid points by augmenting an image of all ones.
            # Pixels that are shifted out of the image boundary will become zero.
            invalid_pts_mask = det_augmenter(image=np.ones(image.shape[:2] + (1,), np.uint8)).squeeze(2)

            # Expand the augmented integer map back into a list of binary masks.
            aug_masks = self.expand_masks(aug_masks.get_arr(), num_instances)

            # Filter the list of augmented masks.
            aug_masks = [mask for mask, is_bm in zip(aug_masks, is_binary_mask)]

            # Return the augmented image and its corresponding augmented masks.
            return aug_image, aug_masks

        # If no masks are provided, just augment the image and return a mask of valid points.
        else:
            # Create a dummy mask to pass to the augmenter.
            masks = [SegmentationMapsOnImage(np.ones(image.shape[:2], np.bool_), shape=image.shape[:2])]
            # Augment the image and the dummy mask.
            aug_image, invalid_pts_mask = det_augmenter(image=image, segmentation_maps=masks)
            # Return the augmented image and a boolean mask where False indicates pixels that are now outside the original image area.
            return aug_image, invalid_pts_mask.get_arr() == 0
