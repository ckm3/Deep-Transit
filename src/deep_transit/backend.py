from .model import YOLOv3
from ._utils import load_model, predict_bboxes, non_max_suppression
import torch
from torchvision import transforms
from . import config

class PytorchBackend:
    def __init__(self, device_str) -> None:
        torch.set_flush_denormal(True)  # Fixing a bug caused by Intel CPU
        self.trans = config.data_transforms()
        if device_str == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_str

    def load_model(self, model_path):
        model = YOLOv3().to(self.device)
        self.model, self.model_config = load_model(model_path, model)
        self.model.eval()

    def inference(self, input, nms_iou_threshold, confidence_threshold):
        if nms_iou_threshold is None:
            nms_iou_threshold = self.model_config['nms_iou_threshold']
        if confidence_threshold is None:
            confidence_threshold = self.model_config['confidence_threshold']
        
        return predict_bboxes(input,
                              model=self.model,
                              iou_threshold=nms_iou_threshold,
                              threshold=confidence_threshold,
                              anchors=self.model_config['anchors'],
                              device_str=self.device
                              )

    def nms(self, input, nms_iou_threshold, confidence_threshold):
        if nms_iou_threshold is None:
            nms_iou_threshold = self.model_config['nms_iou_threshold']
        if confidence_threshold is None:
            confidence_threshold = self.model_config['confidence_threshold']
        return non_max_suppression(input, iou_threshold=nms_iou_threshold,  threshold=confidence_threshold)
