# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Import necessary libraries.
import copy
from collections import defaultdict
from pathlib import Path

import torch
import torch.utils.data

# Import project-specific utilities.
import util.misc as utils
from util.box_ops import generalized_box_iou


class RefExpEvaluator(object):
    """
    A class to evaluate referring expression detection results.
    It calculates Precision@k for different IoU thresholds.
    """

    def __init__(self, refexp_gt, iou_types, k=(1, 5, 10), thresh_iou=0.5):
        """
        Initializes the evaluator.

        Args:
            refexp_gt: The ground truth data, typically a REFER object.
            iou_types: The types of IoU to consider (not directly used here but common in other evaluators).
            k (tuple): A tuple of integers for which to calculate Precision@k (e.g., top 1, 5, 10 predictions).
            thresh_iou (float): The IoU threshold to consider a prediction correct.
        """
        # Ensure k is a list or tuple.
        assert isinstance(k, (list, tuple))
        # Make a deep copy of the ground truth to avoid modifying the original object.
        refexp_gt = copy.deepcopy(refexp_gt)
        self.refexp_gt = refexp_gt
        self.iou_types = iou_types
        # Get the list of all image IDs from the ground truth.
        self.img_ids = self.refexp_gt.imgs.keys()
        # A dictionary to store model predictions, keyed by image ID.
        self.predictions = {}
        # Store the k values for Precision@k calculation.
        self.k = k
        # Store the IoU threshold.
        self.thresh_iou = thresh_iou

    def accumulate(self):
        """
        A placeholder method, often used for accumulating stats over time. Not implemented here.
        """
        pass

    def update(self, predictions):
        """
        Updates the evaluator's internal predictions dictionary with new results from the model.
        """
        self.predictions.update(predictions)

    def synchronize_between_processes(self):
        """
        In a distributed (multi-GPU) setting, this function gathers predictions from all processes
        and merges them into a single dictionary on the main process.
        """
        # Use the utility function to gather prediction dictionaries from all processes.
        all_predictions = utils.all_gather(self.predictions)
        merged_predictions = {}
        # Iterate through the list of dictionaries and merge them.
        for p in all_predictions:
            merged_predictions.update(p)
        # Replace the local predictions with the complete, merged set.
        self.predictions = merged_predictions

    def summarize(self):
        """
        Calculates and prints the final evaluation metrics (Precision@k).
        This method should only be run on the main process after all predictions are synchronized.
        """
        # Ensure this part only runs on the main process to avoid duplicate calculations and printing.
        if utils.is_main_process():
            # Initialize dictionaries to store scores and counts for each dataset.
            dataset2score = {
                "refcoco": {k: 0.0 for k in self.k},
                "refcoco+": {k: 0.0 for k in self.k},
                "refcocog": {k: 0.0 for k in self.k},
            }
            dataset2count = {"refcoco": 0.0, "refcoco+": 0.0, "refcocog": 0.0}

            # Iterate over every image ID in the ground truth test set.
            for image_id in self.img_ids:
                # Get the ground truth annotation ID for the current image.
                ann_ids = self.refexp_gt.getAnnIds(imgIds=image_id)
                assert len(ann_ids) == 1, "Each image should have exactly one referring expression annotation."
                # Load image metadata, which includes the dataset name (e.g., 'refcoco').
                img_info = self.refexp_gt.loadImgs(image_id)[0]

                # Load the ground truth annotation (which contains the target bounding box).
                target = self.refexp_gt.loadAnns(ann_ids[0])
                # Get the model's prediction for this image.
                prediction = self.predictions[image_id]
                assert prediction is not None, "Prediction not found for image."

                # Sort the predicted boxes by their confidence scores in descending order.
                sorted_scores_boxes = sorted(
                    zip(prediction["scores"].tolist(), prediction["boxes"].tolist()), reverse=True
                )
                # Unzip the sorted scores and boxes.
                sorted_scores, sorted_boxes = zip(*sorted_scores_boxes)
                # Convert the list of boxes back into a single tensor.
                sorted_boxes = torch.cat([torch.as_tensor(x).view(1, 4) for x in sorted_boxes])

                # Get the ground truth bounding box in [x, y, width, height] format.
                target_bbox = target[0]["bbox"]
                # Convert the ground truth box to [x1, y1, x2, y2] format.
                converted_bbox = [
                    target_bbox[0],
                    target_bbox[1],
                    target_bbox[2] + target_bbox[0],
                    target_bbox[3] + target_bbox[1],
                ]
                # Calculate the Generalized IoU between all sorted predicted boxes and the single ground truth box.
                giou = generalized_box_iou(sorted_boxes, torch.as_tensor(converted_bbox).view(-1, 4))

                # Check for a correct prediction within the top k boxes.
                for k in self.k:
                    # If the maximum IoU among the top k predictions is above the threshold...
                    if max(giou[:k]) >= self.thresh_iou:
                        # ...count it as a correct prediction for that k.
                        dataset2score[img_info["dataset_name"]][k] += 1.0
                # Increment the total number of samples for this dataset.
                dataset2count[img_info["dataset_name"]] += 1.0

            # Calculate the final precision scores by dividing the correct counts by the total counts.
            for key, value in dataset2score.items():
                for k in self.k:
                    try:
                        value[k] /= dataset2count[key]
                    except ZeroDivisionError:
                        # Handle cases where a dataset might have zero samples.
                        pass

            # Format and print the results.
            results = {}
            for key, value in dataset2score.items():
                results[key] = sorted([v for k, v in value.items()])
                print(f" Dataset: {key} - Precision @ 1, 5, 10: {results[key]} \n")

            return results
        # If not the main process, return None.
        return None
