import torch.utils.data
import torchvision

from .rsvg import build as build_rsvg
from .rsvg_mm import build as build_rsvg_mm


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(dataset_file: str, image_set: str, args):
    if dataset_file == 'rsvg':
        return build_rsvg(image_set, args)
    if dataset_file == 'rsvg_mm':
        return build_rsvg_mm(image_set, args)

    raise ValueError(f'dataset {dataset_file} not supported')
