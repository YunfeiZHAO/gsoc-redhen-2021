"""
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
import os
import json
from PIL import Image
import cv2
import numpy as np
import random

import torch
import torch.utils.data
import torchvision

import datasets.transforms as T
from typing import Any, Callable, Optional, Tuple, List

from pycocotools.coco import COCO  # we set the data format as COCO like
from pycocotools import mask as coco_mask

import argparse


class Chalearn(torchvision.datasets.VisionDataset):
    def __init__(self, root, ann_file, transforms, return_masks):
        super(IDDDetection, self).__init__(root)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

        ann_file = os.path.join(root, 'Annotations', ann_file)
        self.dataset = COCO(ann_file)
        # need to add 'isccrowd' and 'area' for evaluation
        for id, ann in self.dataset.anns.items():
            ann['iscrowd'] = 0
            x, y, w, h = ann['bbox']
            ann['area'] = w * h

        self.categories = self.dataset.dataset['categories']
        self.ids = list(sorted(self.dataset.imgs.keys()))

    def _load_image(self, id: int) -> Image.Image:
        path = self.dataset.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id) -> List[Any]:
        return self.dataset.loadAnns(self.dataset.getAnnIds(id))

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img = self._load_image(image_id)
        target = self._load_target(image_id)
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)  # prepare:  ConvertCocoPolysToMask
        if self._transforms is not None:  # _transforms:  make_coco_transforms
            img, target = self._transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.ids)

    def __repr__(self) -> str:
        head = 'idd dataset'
        body = ["Number of datapoints: {}".format(len(self.ids))]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def extra_repr(self) -> str:
        return ""

    def show_image(self, idx):
        """Need annotation txt file and image file path"""
        image, target = self.__getitem__(self, idx)
        im_h, im_w, _ = image.size()
        labels_num = target['labels']
        rescale = torch.tensor([[im_w, im_h, im_w, im_h]])
        bboxs = target['boxes'] * rescale
        img = image.permute(1, 2, 0).numpy()
        for i, bboxe in enumerate(bboxs):
            x, y, xm, ym = bboxe
            label = class_name[int(labels_num[i])]
            plot_one_box((int(x), int(y), int(xm), int(ym)), img, label=label, line_thickness=3)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def test(image, target):
    _, im_h, im_w = image.size()
    labels_num = target['labels']
    rescale = torch.tensor([[im_w, im_h, im_w, im_h]])
    bboxs = target['boxes'] * rescale
    img = piltensor_to_cvnp(image)
    for i, bbox in enumerate(bboxs):
        x, y, w, h = bbox
        label = class_name[int(labels_num[i])]
        # plot_one_box((int(x), int(y), int(xm), int(ym)), img, label=label, line_thickness=3)
        color = [random.randint(0, 255) for _ in range(3)]
        # Start coordinate, here (5, 5)
        # represents the top left corner of rectangle
        start_point = (int(x-w/2), int(y-h/2))

        # Ending coordinate, here (220, 220)
        # represents the bottom right corner of rectangle
        end_point = (int(x+w/2), int(y+h/2))

        # Blue color in BGR
        # color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2

        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        img = np.array(img).copy()
        img = cv2.rectangle(img, start_point, end_point, color, thickness)
        # cv2.rectangle(img, (x, y), (xm, ym), color, thickness=1, lineType=cv2.LINE_AA)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class_name = {1: 'vehicle fallback', 2: 'rider', 3: 'bus', 4: 'car', 5: 'autorickshaw', 6: 'truck',
              7: 'motorcycle', 8: 'person', 9: 'traffic sign', 10: 'animal', 11: 'bicycle',
              12: 'traffic light', 13: 'caravan', 14: 'train', 15: 'trailer'}


def piltensor_to_cvnp(image):
    """change tensor of PIL image to numpy array of oepncv image"""
    tensor_image = image  # it will change int8 to value between [0,1] of type float32

    # change axi and recover scale
    numpy_image = np.moveaxis((tensor_image * 255).clamp(0, 255).numpy(), 0, -1)

    # change to opencv numpy BGR and change data type
    numpy_cv = numpy_image[:, :, ::-1].astype(np.uint8)
    return numpy_cv


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        boxes[:, 2:] += boxes[:, :2]  # change xy
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        # here area we take as the bounding box area
        area = torch.tensor([obj["area"] for obj in anno])

        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_IDD_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.idd_path)
    assert root.exists(), f'provided IDD path {root} does not exist'
    # mode = 'instances'
    file = {
        "train": 'train.json',
        "val": 'val.json',
    }
    dataset = IDDDetection(root, file[image_set], transforms=make_IDD_transforms(image_set), return_masks=args.masks)
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset_file = 'idd'
    args.idd_path = '/home/yunfei/Desktop/IDD_Detection'
    args.output_dir = './run'
    args.resume = './run/idd_new_checkpoint'
    args.masks = False
    idd_val = build('val', args)
    print(idd_val)
    idd_val.__getitem__(0)
    pass