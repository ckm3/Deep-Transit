"""
Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""

from deep_transit import config
import numpy as np
import os
import pandas as pd
import megengine as mge
import megengine.functional as F

from PIL import Image, ImageFile
from megengine.data import dataset, DataLoader
from .utils import (
    cells_to_bboxes,
    iou_width_height as iou,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class YOLODataset(dataset.Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        image_size=416,
        S=config.S,
        transform=None,
    ):
        self.annotations = pd.read_csv(csv_file, comment='#')
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = np.array(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.loc[index, 'labels'])
        bboxes = np.loadtxt(fname=label_path, delimiter=",", ndmin=2).tolist()
        img_path = os.path.join(self.img_dir, self.annotations.loc[index, 'imgs'])
        image = np.array(Image.open(img_path).convert("L"))
        image = np.expand_dims(image, axis=0)

        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [np.zeros((self.num_anchors // 3, S, S, 5)) for S in self.S]
        for box in bboxes:
            iou_anchors = iou(np.array(box[2:4]), self.anchors)
            anchor_indices = np.argsort(-iou_anchors, axis=0)

            x, y, width, height, snr = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1-np.exp(-0.15*snr)
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = np.array(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return image, tuple(targets)


