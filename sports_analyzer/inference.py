"""Local detection baseline, CPU by default. No implicit weight downloads."""
from pathlib import Path
from time import perf_counter
from .annotations.yolo import Box

class Detector:
    # nms=False uses the NMS-free head of YOLO26; None keeps the default.
    def __init__(self, model_path: Path, device: str | int = 'cpu', nms: bool | None = None):
        if not model_path.is_file():
            raise FileNotFoundError(f'Missing local model weights: {model_path}')
        from ultralytics import YOLO
        self.model = YOLO(str(model_path), task='detect')
        self.device = device
        self.nms = nms

    def predict(self, image_path: Path, imgsz: int, confidence: float):
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if type(imgsz) is not int or imgsz <= 0 or not 0 <= confidence <= 1:
            raise ValueError('Invalid image size or confidence')
        start = perf_counter()
        result = self.model.predict(source=str(image_path), device=self.device, imgsz=imgsz,
                                    conf=confidence, nms=self.nms, rect=True, save=False, verbose=False)[0]
        elapsed_ms = (perf_counter()-start)*1000
        if result.boxes is None:
            raise RuntimeError('Detection model did not return a boxes container')
        boxes = [Box(result.names[int(cls)], tuple(map(float, xyxy)), float(score))
                 for xyxy, cls, score in zip(result.boxes.xyxy.cpu().tolist(),
                                            result.boxes.cls.cpu().tolist(),
                                            result.boxes.conf.cpu().tolist())]
        return boxes, {'predict_call_ms': elapsed_ms, **result.speed}
