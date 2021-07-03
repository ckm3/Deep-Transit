import config
import torch
import torch.optim as optim

from model import YOLOv3
from tqdm.autonotebook import tqdm

from loss import YoloLoss
from utils import (
    load_checkpoint,
    get_loaders,
    cells_to_bboxes,
    non_max_suppression as nms,
    plot_image,
    get_evaluation_bboxes,
)


def show_test_ground_truth():
    model = YOLOv3().to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    # loss_fn = YoloLoss()
    # scaler = torch.cuda.amp.GradScaler()

    # S = [13, 26, 52]
    scaled_anchors = (
            torch.tensor(config.ANCHORS)
            * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.DATASET + "/transit_train.csv", test_csv_path=config.DATASET + "/transit_test.csv"
    )

    load_checkpoint(
        config.CHECKPOINT_FILE + '.tar_35', model, optimizer, config.LEARNING_RATE
    )

    for x, y in test_loader:
        boxes = []
        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            # print(anchor.shape)
            # print(y[i].shape)
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]
        boxes = nms(boxes, iou_threshold=1, threshold=0.1, box_format="midpoint")
        # print(boxes)
        plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes)


def main():
    model = YOLOv3().to(config.DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    train_loader, validation_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.DATASET + "/transit_train.csv", validation_csv_path=config.DATASET + "/transit_val.csv",
        test_csv_path=config.DATASET + "/transit_8examples.csv"
    )

    _ = load_checkpoint(
        config.CHECKPOINT_FILE + '.tar_45', model, optimizer, config.LEARNING_RATE
    )

    pred_boxes, true_boxes = get_evaluation_bboxes(
        test_loader,
        model,
        iou_threshold=config.NMS_IOU_THRESH,
        anchors=config.ANCHORS,
        threshold=config.CONF_THRESHOLD,
    )

    images_list = []
    for x, _ in test_loader:
        for xs in x:
            images_list.append(xs)

    import pandas as pd
    df = pd.DataFrame(pred_boxes)

    gp = df.groupby(0)
    for index, group in gp:
        image = images_list[index]
        plot_image(image.permute(1, 2, 0), group.values[:, 1:])


if __name__ == '__main__':
    main()
    # show_test_ground_truth()
