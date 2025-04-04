from ultralytics import YOLO

class YOLOModel:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def track(self, frame):
        return self.model.track(frame, persist=True)

class YOLOModelFactory:
    @staticmethod
    def create(model_path: str):
        return YOLOModel(model_path)
