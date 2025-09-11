# A variable to store the author's name.
__author__ = "licheng"

"""
This interface provides access to four datasets:
1) refclef
2) refcoco
3) refcoco+
4) refcocog
split by unc and google

The following API functions are defined:
REFER      - REFER api class
getRefIds  - get ref ids that satisfy given filter conditions.
getAnnIds  - get ann ids that satisfy given filter conditions.
getImgIds  - get image ids that satisfy given filter conditions.
getCatIds  - get category ids that satisfy given filter conditions.
loadRefs   - load refs with the specified ref ids.
loadAnns   - load anns with the specified ann ids.
loadImgs   - load images with the specified image ids.
loadCats   - load category names with the specified category ids.
getRefBox  - get ref's bounding box [x, y, w, h] given the ref_id
showRef    - show image, segmentation or box of the referred object with the ref
getMask    - get mask and area of the referred object given ref
showMask   - show mask of the referred object given ref
"""

# Import necessary system and utility libraries.
import sys
import os.path as osp
import json
import pickle
import time
import itertools
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from pprint import pprint
import numpy as np

# Import the mask utilities from pycocotools for handling segmentation masks.
from pycocotools import mask


class REFER:
    """
    The main API class for interacting with referring expression datasets.
    It loads and indexes the dataset annotations for efficient access.
    """

    def __init__(self, data_root, dataset="refcoco", splitBy="unc"):
        """
        Initializes the REFER API object.

        Args:
            data_root (str): The root directory where datasets are stored.
            dataset (str): The name of the dataset to load (e.g., 'refcoco').
            splitBy (str): The split authority (e.g., 'unc', 'google').
        """
        # provide data_root folder which contains refclef, refcoco, refcoco+ and refcocog
        # also provide dataset name and splitBy information
        # e.g., dataset = 'refcoco', splitBy = 'unc'
        print("loading dataset %s into memory..." % dataset)
        # Set up directory paths based on the provided data_root and dataset name.
        self.ROOT_DIR = osp.abspath(osp.dirname(__file__))
        self.DATA_DIR = osp.join(data_root, dataset)  # e.g., coco/refcoco
        # Determine the image directory based on the dataset type.
        if dataset in ["refcoco", "refcoco+", "refcocog"]:
            self.IMAGE_DIR = osp.join(data_root, "train2014")
        elif dataset == "refclef":
            self.IMAGE_DIR = osp.join(data_root, "saiapr_tc-12")
        else:
            # If the dataset name is not recognized, print an error and exit.
            print("No refer dataset is called [%s]" % dataset)
            sys.exit()

        # load refs from data/dataset/refs(dataset).json
        tic = time.time()
        # Construct the path to the pre-processed reference file (a pickled Python object).
        ref_file = osp.join(self.DATA_DIR, "refs(" + splitBy + ").p")
        # Initialize the main data dictionary.
        self.data = {}
        self.data["dataset"] = dataset

        # Load the pickled reference data. This contains the referring expressions and their links to images/annotations.
        self.data["refs"] = pickle.load(open(ref_file, "rb"), fix_imports=True)

        # load annotations from data/dataset/instances.json
        # This file contains the standard COCO-style annotations.
        instances_file = osp.join(self.DATA_DIR, "instances.json")
        instances = json.load(open(instances_file, "r"))  # e.g., coco/refcoco/instances.json
        # Load image metadata (file names, dimensions, etc.).
        self.data["images"] = instances["images"]
        # Load object annotations (segmentations, bounding boxes, etc.).
        self.data["annotations"] = instances["annotations"]
        # Load category information (names, supercategories).
        self.data["categories"] = instances["categories"]

        # Call the method to create efficient look-up tables (indexes).
        self.createIndex()
        print("DONE (t=%.2fs)" % (time.time() - tic))

    def createIndex(self):
        """
        Creates a set of dictionaries that map various IDs to their corresponding data.
        This pre-processing step allows for very fast data retrieval.
        """
        # create sets of mapping
        # 1)  Refs: 	 	{ref_id: ref}
        # 2)  Anns: 	 	{ann_id: ann}
        # 3)  Imgs:		 	{image_id: image}
        # 4)  Cats: 	 	{category_id: category_name}
        # 5)  Sents:     	{sent_id: sent}
        # 6)  imgToRefs: 	{image_id: list_of_refs}
        # 7)  imgToAnns: 	{image_id: list_of_anns}
        # 8)  refToAnn:  	{ref_id: ann}
        # 9)  annToRef:  	{ann_id: ref}
        # 10) catToRefs: 	{category_id: list_of_refs}
        # 11) sentToRef: 	{sent_id: ref}
        # 12) sentToTokens: {sent_id: list_of_tokens}
        print("creating index...")
        # fetch info from instances
        Anns, Imgs, Cats, imgToAnns = {}, {}, {}, {}
        # Index annotations by their ID and group them by image ID.
        for ann in self.data["annotations"]:
            Anns[ann["id"]] = ann
            imgToAnns[ann["image_id"]] = imgToAnns.get(ann["image_id"], []) + [ann]
        # Index images by their ID.
        for img in self.data["images"]:
            Imgs[img["id"]] = img
        # Index categories by their ID.
        for cat in self.data["categories"]:
            Cats[cat["id"]] = cat["name"]

        # fetch info from refs
        Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
        Sents, sentToRef, sentToTokens = {}, {}, {}
        # Index references and sentences, creating all the necessary cross-mappings.
        for ref in self.data["refs"]:
            # ids
            ref_id = ref["ref_id"]
            ann_id = ref["ann_id"]
            category_id = ref["category_id"]
            image_id = ref["image_id"]

            # add mapping related to ref
            Refs[ref_id] = ref
            imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]
            catToRefs[category_id] = catToRefs.get(category_id, []) + [ref]
            refToAnn[ref_id] = Anns[ann_id]
            annToRef[ann_id] = ref

            # add mapping of sent
            for sent in ref["sentences"]:
                Sents[sent["sent_id"]] = sent
                sentToRef[sent["sent_id"]] = ref
                sentToTokens[sent["sent_id"]] = sent["tokens"]

        # Store the created indexes as class members for easy access.
        self.Refs = Refs
        self.Anns = Anns
        self.Imgs = Imgs
        self.Cats = Cats
        self.Sents = Sents
        self.imgToRefs = imgToRefs
        self.imgToAnns = imgToAnns
        self.refToAnn = refToAnn
        self.annToRef = annToRef
        self.catToRefs = catToRefs
        self.sentToRef = sentToRef
        self.sentToTokens = sentToTokens
        print("index created.")

    def getRefIds(self, image_ids=[], cat_ids=[], ref_ids=[], split=""):
        """
        Get reference IDs that satisfy the given filter conditions.
        """
        # Ensure inputs are lists.
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        # If no filters are provided, return all reference IDs.
        if len(image_ids) == len(cat_ids) == len(ref_ids) == len(split) == 0:
            refs = self.data["refs"]
        else:
            # Apply filters sequentially.
            if not len(image_ids) == 0:
                # Use the pre-computed index for fast lookup.
                refs = [self.imgToRefs[image_id] for image_id in image_ids]
            else:
                refs = self.data["refs"]
            if not len(cat_ids) == 0:
                refs = [ref for ref in refs if ref["category_id"] in cat_ids]
            if not len(ref_ids) == 0:
                refs = [ref for ref in refs if ref["ref_id"] in ref_ids]
            if not len(split) == 0:
                # Filter by data split (e.g., 'train', 'val', 'testA').
                if split in ["testA", "testB", "testC"]:
                    refs = [ref for ref in refs if split[-1] in ref["split"]]  # we also consider testAB, testBC, ...
                elif split in ["testAB", "testBC", "testAC"]:
                    refs = [ref for ref in refs if ref["split"] == split]  # rarely used I guess...
                elif split == "test":
                    refs = [ref for ref in refs if "test" in ref["split"]]
                elif split == "train" or split == "val":
                    refs = [ref for ref in refs if ref["split"] == split]
                else:
                    print("No such split [%s]" % split)
                    sys.exit()
        # Return a list of the final filtered reference IDs.
        ref_ids = [ref["ref_id"] for ref in refs]
        return ref_ids

    def getAnnIds(self, image_ids=[], cat_ids=[], ref_ids=[]):
        """
        Get annotation IDs that satisfy the given filter conditions.
        """
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        # If no filters, return all annotation IDs.
        if len(image_ids) == len(cat_ids) == len(ref_ids) == 0:
            ann_ids = [ann["id"] for ann in self.data["annotations"]]
        else:
            # Apply filters sequentially.
            if not len(image_ids) == 0:
                lists = [
                    self.imgToAnns[image_id] for image_id in image_ids if image_id in self.imgToAnns
                ]  # list of [anns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.data["annotations"]
            if not len(cat_ids) == 0:
                anns = [ann for ann in anns if ann["category_id"] in cat_ids]
            ann_ids = [ann["id"] for ann in anns]
            if not len(ref_ids) == 0:
                # Intersect with annotations linked to the given reference IDs.
                ids = set(ann_ids).intersection(set([self.Refs[ref_id]["ann_id"] for ref_id in ref_ids]))
        return ann_ids

    def getImgIds(self, ref_ids=[]):
        """
        Get image IDs associated with the given reference IDs.
        """
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if not len(ref_ids) == 0:
            # Use the index to find image IDs from reference IDs.
            image_ids = list(set([self.Refs[ref_id]["image_id"] for ref_id in ref_ids]))
        else:
            # If no ref_ids are given, return all image IDs.
            image_ids = self.Imgs.keys()
        return image_ids

    def getCatIds(self):
        """
        Get all category IDs in the dataset.
        """
        return self.Cats.keys()

    def loadRefs(self, ref_ids=[]):
        """
        Load full reference data for the given reference IDs.
        """
        if type(ref_ids) == list:
            return [self.Refs[ref_id] for ref_id in ref_ids]
        elif type(ref_ids) == int:
            return [self.Refs[ref_ids]]

    def loadAnns(self, ann_ids=[]):
        """
        Load full annotation data for the given annotation IDs.
        """
        if type(ann_ids) == list:
            return [self.Anns[ann_id] for ann_id in ann_ids]
        elif type(ann_ids) == int:
            return [self.Anns[ann_ids]]

    def loadImgs(self, image_ids=[]):
        """
        Load full image data for the given image IDs.
        """
        if type(image_ids) == list:
            return [self.Imgs[image_id] for image_id in image_ids]
        elif type(image_ids) == int:
            return [self.Imgs[image_ids]]

    def loadCats(self, cat_ids=[]):
        """
        Load category names for the given category IDs.
        """
        if type(cat_ids) == list:
            return [self.Cats[cat_id] for cat_id in cat_ids]
        elif type(cat_ids) == int:
            return [self.Cats[cat_ids]]

    def getRefBox(self, ref_id):
        """
        Get the bounding box [x, y, w, h] for a given reference ID.
        """
        ref = self.Refs[ref_id]
        # Use the refToAnn index to find the corresponding annotation.
        ann = self.refToAnn[ref_id]
        return ann["bbox"]  # [x, y, w, h]

    def showRef(self, ref, seg_box="seg"):
        """
        Display an image and overlay the referred object's segmentation or bounding box.
        """
        ax = plt.gca()
        # show image
        image = self.Imgs[ref["image_id"]]
        I = io.imread(osp.join(self.IMAGE_DIR, image["file_name"]))
        ax.imshow(I)
        # show refer expression
        for sid, sent in enumerate(ref["sentences"]):
            print("%s. %s" % (sid + 1, sent["sent"]))
        # show segmentations
        if seg_box == "seg":
            ann_id = ref["ann_id"]
            ann = self.Anns[ann_id]
            polygons = []
            color = []
            c = "none"
            if type(ann["segmentation"][0]) == list:
                # This handles polygon format segmentation, common in refcoco*.
                for seg in ann["segmentation"]:
                    poly = np.array(seg).reshape((len(seg) // 2, 2))
                    polygons.append(Polygon(poly, True, alpha=0.4))
                    color.append(c)
                # Add the polygon patches to the plot.
                p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 1, 0, 0), linewidths=3, alpha=1)
                ax.add_collection(p)  # thick yellow polygon
                p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 0, 0, 0), linewidths=1, alpha=1)
                ax.add_collection(p)  # thin red polygon
            else:
                # This handles RLE (Run-Length Encoding) format segmentation.
                rle = ann["segmentation"]
                m = mask.decode(rle)
                img = np.ones((m.shape[0], m.shape[1], 3))
                color_mask = np.array([2.0, 166.0, 101.0]) / 255
                for i in range(3):
                    img[:, :, i] = color_mask[i]
                ax.imshow(np.dstack((img, m * 0.5)))
        # show bounding-box
        elif seg_box == "box":
            ann_id = ref["ann_id"]
            ann = self.Anns[ann_id]
            bbox = self.getRefBox(ref["ref_id"])
            # Create a rectangle patch and add it to the plot.
            box_plot = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor="green", linewidth=3)
            ax.add_patch(box_plot)

    def getMask(self, ref):
        """
        Get the binary segmentation mask for a given reference.
        """
        # return mask, area and mask-center
        ann = self.refToAnn[ref["ref_id"]]
        image = self.Imgs[ref["image_id"]]
        # Convert polygon format to RLE if necessary.
        if type(ann["segmentation"][0]) == list:  # polygon
            rle = mask.frPyObjects(ann["segmentation"], image["height"], image["width"])
        else:  # It's already in RLE format.
            rle = ann["segmentation"]

        # Decode the RLE to get a binary mask.
        m = mask.decode(rle)
        # Handle cases where a single annotation has multiple disconnected parts.
        m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
        m = m.astype(np.uint8)  # convert to np.uint8
        # compute area
        area = sum(mask.area(rle))  # should be close to ann['area']
        return {"mask": m, "area": area}

    def showMask(self, ref):
        """
        A simple utility to display the mask of a referred object.
        """
        M = self.getMask(ref)
        msk = M["mask"]
        ax = plt.gca()
        ax.imshow(msk)
