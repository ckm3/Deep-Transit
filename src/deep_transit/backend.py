from .model import YOLOv3
from ._utils import load_model, predict_bboxes, non_max_suppression
import torch
from torchvision import transforms
from . import config

class PytorchBackend:
    def __init__(self) -> None:
        torch.set_flush_denormal(True)  # Fixing a bug caused by Intel CPU
        self.trans = config.data_transforms()


    def load_model(self, device_str, model_path):
        if device_str is not None:
            model = YOLOv3().to(device_str)
        else:
            model = YOLOv3()
        self.model, self.model_config = load_model(model_path, model)
        self.model.eval()

    def inference(self, input, nms_iou_threshold, confidence_threshold, device_str):
        if nms_iou_threshold is None:
            nms_iou_threshold = self.model_config['nms_iou_threshold']
        if confidence_threshold is None:
            confidence_threshold = self.model_config['confidence_threshold']
        
        return predict_bboxes(input,
                              model=self.model,
                              iou_threshold=nms_iou_threshold,
                              threshold=confidence_threshold,
                              anchors=self.model_config['anchors'],
                              device_str=device_str
                              )

    def nms(self, input, nms_iou_threshold, confidence_threshold):
        if nms_iou_threshold is None:
            nms_iou_threshold = self.model_config['nms_iou_threshold']
        if confidence_threshold is None:
            confidence_threshold = self.model_config['confidence_threshold']
        return non_max_suppression(input, iou_threshold=nms_iou_threshold,  threshold=confidence_threshold)
