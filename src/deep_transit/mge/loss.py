"""
Implementation of Yolo Loss Function
"""

import megengine
import megengine.module as nn
import megengine.functional as F
import megengine.functional.loss as loss


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()

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
        no_object_loss = loss.binary_cross_entropy(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #
        
        if F.sum(target[..., 1:5][obj]).item() == 0:
            object_loss = 0
            box_loss = 0
        else:
            anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
            object_loss = loss.binary_cross_entropy(predictions[..., 0][obj], target[..., 0][obj])   # target[..., 0] is the SNR

            # ======================== #
            #   FOR BOX COORDINATES    #
            # ======================== #

            predictions[..., 1:3] = F.sigmoid(predictions[..., 1:3])  # x,y coordinates
            target[..., 3:5] = F.log(
                (1e-16 + target[..., 3:5] / anchors)
            )  # width, height coordinates

            # MSE loss
            box_loss = loss.square_loss(predictions[..., 1:5][obj], target[..., 1:5][obj])


        return (
                self.lambda_box * box_loss
                + self.lambda_obj * object_loss
                + self.lambda_noobj * no_object_loss
        )
