from . import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch
from torchvision.ops import nms
from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm


def warning_on_one_line(message, category, filename, lineno, line=None):
    return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)


def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 : tensor
                width and height of the first bounding boxes
        boxes2 : tensor
                width and height of the second bounding boxes
    Returns:
        tensor
        Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
            boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


def intersection_over_union(box1, box2, box_format="midpoint", eps=1e-7):
    box2 = box2.T
    box1 = box1.T

    if box_format == "corner":  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    elif box_format == "midpoint":  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    
    return iou  # IoU


def old_intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """

    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds : tensor
                    Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels : tensor
                    Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format : str
                    midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor
        Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="midpoint"):
    """
    Using the NMS implemented by torchvision
    Parameters
    ----------
    bboxes : list
            list of lists containing all bboxes with each bboxes
            specified as [prob_score, x1, y1, x2, y2]
    iou_threshold : float
                    IoU threshold where predicted bboxes is correct
    threshold : float
                Threshold to remove predicted bboxes before NMS
    box_format : float
                "midpoint" or "corners" used to specify bboxes

    Returns
    -------
    list
    bboxes after performing NMS given a specific IoU threshold
    """
    assert type(bboxes) == list

    bboxes = torch.tensor([box for box in bboxes if box[0] > threshold])
    if len(bboxes) == 0:
        return []

    if box_format == "midpoint":
        box1_x1 = bboxes[..., 1:2] - bboxes[..., 3:4] / 2
        box1_y1 = bboxes[..., 2:3] - bboxes[..., 4:5] / 2
        box1_x2 = bboxes[..., 1:2] + bboxes[..., 3:4] / 2
        box1_y2 = bboxes[..., 2:3] + bboxes[..., 4:5] / 2

        mid_bboxes = torch.cat((bboxes[..., 0:1], box1_x1, box1_y1, box1_x2, box1_y2), dim=1)
    else:
        mid_bboxes = bboxes
    index_tensor = nms(mid_bboxes[..., 1:], mid_bboxes[..., 0], iou_threshold)
    return bboxes[index_tensor].tolist()


def average_precision(
        pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint"
):
    """
    This function calculates average precision (AP)
    Parameters
    ----------
    pred_boxes : list
                list of lists containing all bboxes with each bboxes
                specified as [train_idx, confidence, x, y, w, h]
    true_boxes : list
                Similar as pred_boxes except all the correct ones
    iou_threshold : float
                    Threshold where predicted bboxes is correct
    box_format : str
                "midpoint" or "corners" used to specify bboxes

    Returns
    -------
    float
    AP value given a specific IoU threshold
    """
    # used for numerical stability later on
    epsilon = 1e-6

    # find the amount of bboxes for each training example
    # Counter here finds how many ground truth bboxes we get
    # for each training example, so let's say img 0 has 3,
    # img 1 has 5 then we will obtain a dictionary with:
    # amount_bboxes = {0:3, 1:5}
    amount_bboxes = Counter([gt[0] for gt in true_boxes])

    # We then go through each key, val in this dictionary
    # and convert to the following (w.r.t same example):
    # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
    for key, val in amount_bboxes.items():
        amount_bboxes[key] = torch.zeros(val)

    # sort by box probabilities which is index 2
    pred_boxes.sort(key=lambda x: x[1], reverse=True)
    TP = torch.zeros((len(pred_boxes)))
    FP = torch.zeros((len(pred_boxes)))
    total_true_bboxes = len(true_boxes)

    # If none exists then we can safely skip
    if total_true_bboxes == 0:
        return 0

    for detection_idx, detection in enumerate(pred_boxes):
        # Only take out the ground_truths that have the same
        # training idx as detection
        ground_truth_img = [
            bbox for bbox in true_boxes if bbox[0] == detection[0]
        ]

        # num_gts = len(ground_truth_img)
        best_iou = 0

        for idx, gt in enumerate(ground_truth_img):
            iou = intersection_over_union(
                torch.tensor(detection[2:]),
                torch.tensor(gt[2:]),
                box_format=box_format,
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou > iou_threshold:
            # only detect ground truth detection once
            if amount_bboxes[detection[0]][best_gt_idx] == 0:
                # true positive and add this bounding box to seen
                TP[detection_idx] = 1
                amount_bboxes[detection[0]][best_gt_idx] = 1
            else:
                FP[detection_idx] = 1

        # if IOU is lower then the detection is a false positive
        else:
            FP[detection_idx] = 1

    TP_cumsum = torch.cumsum(TP, dim=0)
    FP_cumsum = torch.cumsum(FP, dim=0)
    recalls = TP_cumsum / (total_true_bboxes + epsilon)
    precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
    precisions = torch.cat((torch.tensor([1]), precisions))
    recalls = torch.cat((torch.tensor([0]), recalls))

    # torch.trapz for numerical integration
    # ap.append(torch.trapz(precisions, recalls))

    return torch.trapz(precisions, recalls).float()


def save_PR_curve(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", figure_path=None):
    """
    Save PR curve to a figure for checking performance conveniently.
    Parameters
    ----------
    pred_boxes : list
                list of lists containing all bboxes with each bboxes
                specified as [train_idx, confidence, x, y, w, h]
    true_boxes : list
                Similar as pred_boxes except all the correct ones
    iou_threshold : float
                    Threshold where predicted bboxes is correct
    box_format : str
                "midpoint" or "corners" used to specify bboxes
    """
    # used for numerical stability later on
    epsilon = 1e-6

    # find the amount of bboxes for each training example
    # Counter here finds how many ground truth bboxes we get
    # for each training example, so let's say img 0 has 3,
    # img 1 has 5 then we will obtain a dictionary with:
    # amount_bboxes = {0:3, 1:5}
    amount_bboxes = Counter([gt[0] for gt in true_boxes])

    # We then go through each key, val in this dictionary
    # and convert to the following (w.r.t same example):
    # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
    for key, val in amount_bboxes.items():
        amount_bboxes[key] = torch.zeros(val)

    # sort by box probabilities which is index 2
    pred_boxes.sort(key=lambda x: x[1], reverse=True)
    TP = torch.zeros((len(pred_boxes)))
    FP = torch.zeros((len(pred_boxes)))
    total_true_bboxes = len(true_boxes)

    # If none exists then we can safely skip
    if total_true_bboxes == 0:
        return 0

    for detection_idx, detection in enumerate(pred_boxes):
        # Only take out the ground_truths that have the same
        # training idx as detection
        ground_truth_img = [
            bbox for bbox in true_boxes if bbox[0] == detection[0]
        ]

        best_iou = 0

        for idx, gt in enumerate(ground_truth_img):
            iou = intersection_over_union(
                torch.tensor(detection[2:]),
                torch.tensor(gt[2:]),
                box_format=box_format,
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou > iou_threshold:
            # only detect ground truth detection once
            if amount_bboxes[detection[0]][best_gt_idx] == 0:
                # true positive and add this bounding box to seen
                TP[detection_idx] = 1
                amount_bboxes[detection[0]][best_gt_idx] = 1
            else:
                FP[detection_idx] = 1

        # if IOU is lower then the detection is a false positive
        else:
            FP[detection_idx] = 1

    TP_cumsum = torch.cumsum(TP, dim=0)
    FP_cumsum = torch.cumsum(FP, dim=0)
    recalls = TP_cumsum / (total_true_bboxes + epsilon)
    precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
    precisions = torch.cat((torch.tensor([1]), precisions))
    recalls = torch.cat((torch.tensor([0]), recalls))

    with plt.rc_context({'backend': 'agg'}):
        plt.figure()
        plt.plot(recalls, precisions, 'k')
        plt.savefig(figure_path)
        plt.close()


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    # cmap = plt.get_cmap("tab20b")
    # colors = [cmap(i) for i in np.linspace(0, 1, len(boxes))]
    im = np.array(image)

    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im.reshape(image.shape[:2]), cmap='binary_r', origin='upper')

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for i, box in enumerate(boxes):
        assert len(box) == 5, "box should contain confidence, x, y, width, height"
        confidence = box[0]
        box = box[1:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor='lime',
            # edgecolor=colors[i],
            facecolor="none",
        )

        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=f"{confidence:.2f}",
            color="white",
            verticalalignment="top",
            bbox={"pad": 0},
        )

    plt.show()


def predict_bboxes(image, model, iou_threshold, threshold, anchors,
                   device_str):
    image = torch.tensor(np.stack(image), device=device_str)
    
    with torch.no_grad():
        predictions = model(image)

    batch_size = image.shape[0]
    bboxes = [[] for _ in range(batch_size)]
    for i in range(3):
        S = predictions[i].shape[2]
        anchor = torch.tensor([*anchors[i]], device=device_str) * S
        boxes_scale_i = cells_to_bboxes(
            predictions[i], anchor, S=S, is_preds=True
        )
        for idx, (box) in enumerate(boxes_scale_i):
            bboxes[idx] += box
            
    nms_boxes = []
    for lc_index in range(batch_size):
        nms_boxes.append(non_max_suppression(
            bboxes[lc_index],
            iou_threshold=iou_threshold,
            threshold=threshold,
            box_format="midpoint",
        ))
    return nms_boxes


def get_evaluation_bboxes(
        loader,
        model,
        iou_threshold,
        anchors,
        threshold,
        box_format="midpoint",
        device=config.DEVICE,
):
    # make sure models is in eval before get bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        # x = x.float().to(device)
        x = x.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]], device=device) * S
            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # we just want one bbox for each label, not one for each scale
        true_bboxes = cells_to_bboxes(
            labels[2], anchor, S=S, is_preds=False
        )

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[0] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Scales the predictions coming from the models to
    be relative to the entire image such that they for example later
    can be plotted or evaluated.
    Parameters
    ----------
    predictions : tensor
                The size is (N, 3, S, S, 5)
    anchors : np.ndarray
            The anchors used for the predictions
    S : int
        The number of cells the image is divided in on the width (and height)
    is_preds : bool
            Whether the input is predictions or the true bounding boxes
    Returns
    -------
    converted_bboxes : list
                    the converted boxes of sizes (N, num_anchors, S, S, 1+5) with
                    object confidence, bounding box coordinates
    """
    batch_size = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, num_anchors, 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
    else:
        scores = predictions[..., 0:1]

    cell_indices = (
        torch.arange(S)
            .repeat(predictions.shape[0], num_anchors, S, 1)
            .unsqueeze(-1)
            .to(predictions.device)
    )
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((scores, x, y, w_h), dim=-1).reshape(batch_size, num_anchors * S * S, 5)
    return converted_bboxes.tolist()


def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def save_checkpoint(model, optimizer, epoch, lr_scheduler, file_path="checkpoint.pth.tar"):
    tqdm.write("=> Saving checkpoint")
    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler_state": lr_scheduler.state_dict(),
        "anchors" : config.ANCHORS,
        "nms_iou_threshold" : config.NMS_IOU_THRESH,
        "confidence_threshold" : config.CONF_THRESHOLD,
    }
    torch.save(checkpoint, file_path)


def load_checkpoint(checkpoint_file, model, optimizer, lr, lr_scheduler):
    tqdm.write("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
    epoch = checkpoint["epoch"]

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    # for param_group in optimizer.param_groups:
    #     param_group["lr"] = lr
    return epoch

def load_model(model_file, model, device_str=None):
    tqdm.write(f"Loading Model: {model_file}")
    if device_str is not None:
        ckpt = torch.load(model_file, map_location=device_str)
    else:
        ckpt = torch.load(model_file)
    if 'state_dict' in ckpt.keys():
        model.load_state_dict(ckpt['state_dict'])
        return model, {key:ckpt[key] for key in ckpt.keys() if key != 'state_dict'}
    else:
        # compatiable with bare dump
        model.load_state_dict(ckpt)
        return  model, {"anchors" : config.ANCHORS,
                        "nms_iou_threshold" : config.NMS_IOU_THRESH,
                        "confidence_threshold" : config.CONF_THRESHOLD}


def save_checkpoint_to_model(checkpoint_path, model_path):
    """
    Save trained a checkpoint to a model

    Parameters
    ----------
    checkpoint_path : str
    model_path : str
    """
    from . import config
    from ._utils import load_checkpoint
    from .model import YOLOv3
    import torch.optim as optim

    model = YOLOv3().to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, verbose=True,
                                                        cooldown=3)
    _ = load_checkpoint(checkpoint_path, model, optimizer, config.LEARNING_RATE, lr_scheduler)
    print("=> Saving model")
    torch.save({
        'state_dict': model.state_dict(),
        "anchors": config.ANCHORS,
        "nms_iou_threshold": config.NMS_IOU_THRESH,
        "confidence_threshold": config.CONF_THRESHOLD}, model_path)


def export_model_to_onnx(model_file, onnx_file):
    import torch.onnx
    from . import config
    from ._utils import load_checkpoint
    from .model import YOLOv3
    import torch.optim as optim

    # A model class instance (class not shown)
    model = YOLOv3()

    # Load the weights from a file (.pth usually)
    state_dict = torch.load(model_file)

    # Load the weights now into a model net architecture defined by our class
    model.load_state_dict(state_dict)

    # Create the right input shape (e.g. for an image)
    dummy_input = torch.randn(2, 1, 416, 416)

    torch.onnx.export(model, dummy_input, onnx_file)


def get_loaders(train_csv_path, validation_csv_path):
    from .dataset import YOLODataset

    image_size = config.IMAGE_SIZE
    train_dataset = YOLODataset(
        train_csv_path,
        transform=config.data_transforms(),
        S=[image_size // 32, image_size // 16, image_size // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )
    validation_dataset = YOLODataset(
        validation_csv_path,
        transform=config.data_transforms(),
        S=[image_size // 32, image_size // 16, image_size // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )
    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, validation_loader


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
