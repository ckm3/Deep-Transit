from deep_transit import model
from .model import YOLOv3
from tqdm.autonotebook import tqdm
from .utils import (
    get_loaders,
    get_evaluation_bboxes,
    average_precision
)
from .loss import YoloLoss
from  .. import config 

import megengine.optimizer as optim
import megengine as mge
from megengine.autodiff import GradManager
import numpy as np
train_loader, validation_loader = get_loaders(
        train_csv_path=config.DATASET + "/transit_train.csv", validation_csv_path=config.DATASET + "/transit_val.csv",
    )

def train_fn(train_loader, model, optimizer,  loss_fn, scaler, scaled_anchors, gm):
    model.train()
    loop = tqdm(train_loader)
    avg_loss = -1
    for batch_idx, (x, y) in enumerate(loop):
        optimizer.zero_grad()
        with gm:
            out = model(x)
            loss = (
                    loss_fn(out[0], mge.Tensor(y[0]), scaled_anchors[0])
                    + loss_fn(out[1], mge.Tensor(y[1]), scaled_anchors[1])
                    + loss_fn(out[2], mge.Tensor(y[2]), scaled_anchors[2])
            )
            gm.backward(loss)
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

def train(patience=2, cooldown=3, enable_seed_everything=True):

    model = YOLOv3()
    model.train()
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    gm = GradManager().attach(model.parameters())

    loss_fn = YoloLoss()
    scaler = None

    tqdm.write(config.DATASET)
    train_loader, validation_loader = get_loaders(
        train_csv_path=config.DATASET + "/transit_train.csv", validation_csv_path=config.DATASET + "/transit_val.csv",
    )

    epoch_old = 0

    np_s = np.array(config.S).reshape(len(config.S), 1, 1)
    np_s_tile  = np.tile(np_s, (1, sum([len(x) for x in config.ANCHORS]) // 3, 2))
    scaled_anchors = mge.Tensor(np.array(config.ANCHORS) * np_s_tile)


    for epoch in range(epoch_old, config.NUM_EPOCHS + 1):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors, gm)
        mge.save(
                    {
                        "epoch": epoch,
                        "state_dict" : model.state_dict(),
                        "anchors" : config.ANCHORS,
                        "nms_iou_threshold" : config.NMS_IOU_THRESH,
                        "confidence_threshold" : config.CONF_THRESHOLD,
                    },
                    './ckpt_deep_transit_{}.pkl'.format(epoch)
                )
        pred_boxes, true_boxes = get_evaluation_bboxes(
            validation_loader,
            model,
            iou_threshold=config.NMS_IOU_THRESH,
            anchors=config.ANCHORS,
            threshold=config.CONF_THRESHOLD,
        )
        AP50 = average_precision(
                    pred_boxes,
                    true_boxes,
                    iou_threshold=config.MAP_IOU_THRESH,
                    box_format="midpoint",
                )
        AP70 = average_precision(
                    pred_boxes,
                    true_boxes,
                    iou_threshold=0.70,
                    box_format="midpoint",
                )
        AP90 = average_precision(
            pred_boxes,
            true_boxes,
            iou_threshold=0.9,
            box_format="midpoint",
        )
        mAP = (AP50 + AP70 + AP90) / 3
        tqdm.write(f"AP50: {AP50:.3f}, AP750: {AP70:.3f}, AP90: {AP90:.3f}")
        
if __name__ == "__main__":
    train()
