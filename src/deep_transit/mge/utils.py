import megengine as mge
import megengine.functional as F
from .. import config
from megengine.data import DataLoader
from megengine.data.sampler import RandomSampler, SequentialSampler
import numpy as np
import tqdm
from collections import Counter


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
    intersection = np.minimum(boxes1[..., 0], boxes2[..., 0]) *np.minimum(
       boxes1[..., 1], boxes2[..., 1]
    )
    union = (
            boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union

def sigmoid(x):
    return 1/(1 + np.exp(-x))

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
        box_predictions[..., 0:2] = sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = np.exp(box_predictions[..., 2:]) * anchors
        scores = sigmoid(predictions[..., 0:1])
    else:
        scores = predictions[..., 0:1]
    cell_indices = np. expand_dims(np.tile(
        np.arange(S),
        (batch_size, num_anchors, S, 1)), -1)
           
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.transpose(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = np.concatenate((scores, x, y, w_h), axis=-1).reshape(batch_size, num_anchors * S * S, 5)
    return converted_bboxes

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
        sampler=SequentialSampler(train_dataset, batch_size=config.BATCH_SIZE),
        num_workers=config.NUM_WORKERS,
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        sampler=SequentialSampler(validation_dataset, batch_size=config.BATCH_SIZE),
        num_workers=config.NUM_WORKERS,
    )

    return train_loader, validation_loader

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

    bboxes = mge.tensor([box for box in bboxes if box[0] > threshold])
    if len(bboxes) == 0:
        return []

    if box_format == "midpoint":
        box1_x1 = bboxes[..., 1:2] - bboxes[..., 3:4] / 2
        box1_y1 = bboxes[..., 2:3] - bboxes[..., 4:5] / 2
        box1_x2 = bboxes[..., 1:2] + bboxes[..., 3:4] / 2
        box1_y2 = bboxes[..., 2:3] + bboxes[..., 4:5] / 2

        mid_bboxes = F.concat((bboxes[..., 0:1], box1_x1, box1_y1, box1_x2, box1_y2), axis=1)
    else:
        mid_bboxes = bboxes
    index_tensor = F.vision.nms(mid_bboxes[..., 1:], mid_bboxes[..., 0], iou_threshold)
    return bboxes[index_tensor].tolist()


def predict_bboxes(image, model, iou_threshold, threshold, anchors):
    image = np.stack(image)
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=1)
    predictions = model(image)

    batch_size = image.shape[0]
    bboxes = [[] for _ in range(batch_size)]
    for i in range(3):
        S = predictions[i].shape[2]
        anchor = np.array([*anchors[i]]) * S
        boxes_scale_i = cells_to_bboxes(
                predictions[i].numpy().copy(), anchor, S=S, is_preds=True
            )
        for idx, (box) in enumerate(boxes_scale_i):
            bboxes[idx] += box.tolist()
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
):
    # make sure models is in eval before get bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    
    for batch_idx, (x, labels) in enumerate(tqdm.tqdm(loader)):
        predictions = model(x)
        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            S = predictions[i].shape[2]            
            anchor = np.array([*anchors[i]]) * S
            boxes_scale_i = cells_to_bboxes(
                predictions[i].numpy().copy(), anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box.tolist()

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
                    all_true_boxes.append([train_idx] + box.tolist())

            train_idx += 1

    return all_pred_boxes, all_true_boxes

def intersection_over_union(box1, box2, box_format="midpoint", eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is N*4, box2 is N*4
    box2 = box2.T
    box1 = box1.T
    # Get the coordinates of bounding boxes
    if box_format == "corner":  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    elif box_format == "midpoint":  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = np.maximum(np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1),0) * \
            np.maximum(np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1),0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    return iou

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
    amount_bboxes = Counter([gt[0] for gt in true_boxes])

    # We then go through each key, val in this dictionary
    # and convert to the following (w.r.t same example):
    for key, val in amount_bboxes.items():
        amount_bboxes[key] = np.zeros(val)

    # sort by box probabilities which is index 2
    pred_boxes.sort(key=lambda x: x[1], reverse=True)
    TP = np.zeros((len(pred_boxes)))
    FP = np.zeros((len(pred_boxes)))
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
                np.array(detection[2:]),
                np.array(gt[2:]),
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

    TP_cumsum = np.cumsum(TP, axis=0)
    FP_cumsum = np.cumsum(FP, axis=0)
    
    recalls = TP_cumsum / (total_true_bboxes + epsilon)
    precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
    precisions = np.concatenate((np.array([1]), precisions))
    recalls = np.concatenate((np.array([0]), recalls))

    return np.trapz(precisions, recalls)

def load_model(model_file, model):
    tqdm.tqdm.write(f"Loading Model: {model_file} with megengine")
    ckpt = mge.load(model_file)
    model.load_state_dict(ckpt['state_dict'])
    return model, {key:ckpt[key] for key in ckpt.keys() if key != 'state_dict'}
