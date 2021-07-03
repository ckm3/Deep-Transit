import config
import torch
import torch.optim as optim

from model import YOLOv3

from utils import (
    load_checkpoint,
    get_loaders,
    cells_to_bboxes,
    average_precision,
    plot_image,
    get_evaluation_bboxes,
)


def main():
    model = YOLOv3().to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    train_loader, validation_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.DATASET + "/transit_train.csv", validation_csv_path=config.DATASET + "/transit_val.csv",
        test_csv_path=config.DATASET + "/transit_8examples.csv"
    )

    load_checkpoint(
        config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
    )

    pred_boxes, true_boxes = get_evaluation_bboxes(
        test_loader,
        model,
        iou_threshold=config.NMS_IOU_THRESH,
        anchors=config.ANCHORS,
        threshold=config.CONF_THRESHOLD,
    )

    apval = average_precision(
        pred_boxes,
        true_boxes,
        iou_threshold=config.MAP_IOU_THRESH,
        box_format="midpoint",
    )

    print(f"MAP: {apval}")


if __name__ == '__main__':
    main()
