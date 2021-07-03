"""
Main file for training Yolo models on Pascal VOC and COCO dataset
"""

# import os

# os.environ["WANDB_MODE"] = "offline"

from . import config
import torch
import torch.optim as optim

torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)
from .model import YOLOv3
from tqdm.autonotebook import tqdm
from .utils import (
    average_precision,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    get_loaders,
)
from .loss import YoloLoss


# import wandb
#
# wandb.init(project='deep_transit',
#            config=dict(
#                LEARNING_RATE=config.LEARNING_RATE,
#                WEIGHT_DECAY=config.WEIGHT_DECAY,
#                BATCH_SIZE=config.BATCH_SIZE,
#            ))


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader)
    avg_loss = -1
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )
        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                    loss_fn(out[0], y0, scaled_anchors[0])
                    + loss_fn(out[1], y1, scaled_anchors[1])
                    + loss_fn(out[2], y2, scaled_anchors[2])
            )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        loop.set_postfix(loss=loss.item())
        if avg_loss == -1:
            avg_loss = loss
        avg_loss = avg_loss * 0.95 + loss * 0.05
        # wandb.log({"loss": loss.item(), "avg loss": avg_loss.item()})


def train(patience=2, cooldown=3):
    model = YOLOv3().to(config.DEVICE)
    # models = SimplifiedYOLOv3().to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=patience, factor=0.5,
                                                        verbose=True, cooldown=cooldown)
    tqdm.write(config.DATASET)
    train_loader, validation_loader = get_loaders(
        train_csv_path=config.DATASET + "/transit_train.csv", validation_csv_path=config.DATASET + "/transit_val.csv",
    )

    epoch_old = 0
    if config.LOAD_MODEL:
        epoch_old = load_checkpoint(
            config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE, lr_scheduler
        ) + 1

    scaled_anchors = (
            torch.tensor(config.ANCHORS)
            * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, sum([len(x) for x in config.ANCHORS]) // 3, 2)
    ).to(config.DEVICE)

    # wandb.watch(models)

    for epoch in range(epoch_old, config.NUM_EPOCHS + 1):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        if epoch % 5 == 0 and epoch > 0:
            if config.SAVE_MODEL:
                save_checkpoint(model, optimizer, epoch, lr_scheduler,
                                file_path=f"{config.CHECKPOINT_FILE}_{epoch}.tar")

        tqdm.write("On Validation loader:")

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
        AP75 = average_precision(
            pred_boxes,
            true_boxes,
            iou_threshold=0.75,
            box_format="midpoint",
        )
        AP90 = average_precision(
            pred_boxes,
            true_boxes,
            iou_threshold=0.9,
            box_format="midpoint",
        )
        mAP = (AP50 + AP75 + AP90) / 3
        lr_scheduler.step(mAP)
        tqdm.write(f"AP50: {AP50:.3f}, AP75: {AP75:.3f}, AP90: {AP90:.3f}")
        # wandb.log({'epoch': epoch, 'ap50': AP50, 'ap75': AP75, 'ap90': AP90, 'mAP': mAP,
        #            'lr': optimizer.param_groups[0]['lr']})


if __name__ == "__main__":
    train()
