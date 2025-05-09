from ultralytics import YOLO
from Scripts.base_model import BaseModel

class YOLOModel(BaseModel):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.model = YOLO(model_path)

    def track(self, frame):
        return self.model.track(frame, persist=True,verbose=False)

#Explicit handler for YOLOv8 models
class YOLOv8Model(BaseModel):
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        super().__init__(model_path)
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def track(self, frame):
        results = self.model.track(frame, persist=True, verbose=False)

        if self.confidence_threshold > 0:
            results = self._filter_by_confidence(results)

        return results

    def _filter_by_confidence(self, results):
        for result in results:
            boxes = result.boxes
            confidences = boxes.conf

            mask = confidences >= self.confidence_threshold
            result.boxes = boxes[mask]

        return results


class YOLOModelFactory:
    @staticmethod
    def create(model_path: str, confidence_threshold: float = 0.5) -> BaseModel:
        if "v8" in model_path.lower():
            return YOLOv8Model(model_path, confidence_threshold)
        else:
            return YOLOModel(model_path)

