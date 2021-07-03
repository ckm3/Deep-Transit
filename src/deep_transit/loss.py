"""
Implementation of Yolo Loss Function
"""

import torch
import torch.nn as nn


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):
        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] > 0  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #
        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        if target[..., 1:5][obj].nelement() == 0:
            object_loss = 0
            box_loss = 0
        else:
            anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
            object_loss = self.bce(predictions[..., 0][obj], target[..., 0][obj])   # target[..., 0] is the SNR

            # ======================== #
            #   FOR BOX COORDINATES    #
            # ======================== #

            predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates
            target[..., 3:5] = torch.log(
                (1e-16 + target[..., 3:5] / anchors)
            )  # width, height coordinates

            # MSE loss
            box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])


        return (
                self.lambda_box * box_loss
                + self.lambda_obj * object_loss
                + self.lambda_noobj * no_object_loss
        )
