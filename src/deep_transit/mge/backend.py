from .model import YOLOv3
from .utils import load_model, predict_bboxes, non_max_suppression


class MegengineBackend:
    def __init__(self) -> None:
        def bypass(x):return x
        self.trans = bypass
        pass

    def load_model(self, model_path):
        model = YOLOv3()
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
                              anchors=self.model_config['anchors']
                              )

    def nms(self, input, nms_iou_threshold, confidence_threshold):
        if nms_iou_threshold is None:
            nms_iou_threshold = self.model_config['nms_iou_threshold']
        if confidence_threshold is None:
            confidence_threshold = self.model_config['confidence_threshold']
        return non_max_suppression(input, iou_threshold=nms_iou_threshold,  threshold=confidence_threshold)
