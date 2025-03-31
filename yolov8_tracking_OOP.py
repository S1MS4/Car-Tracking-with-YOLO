import sys
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# Singleton
class YOLOModel:
    _instance = None
    def __new__(cls, model_path='yolov8n.pt'):
        if cls._instance is None:
            cls._instance = super(YOLOModel, cls).__new__(cls)
            cls._instance.model = YOLO(model_path)
        return cls._instance

    def track(self, frame):
        return self.model.track(frame, persist=True)

# Abstraction :)
class TrackerBase:
    def process_frame(self, frame):
        raise NotImplementedError("Implement method")


class YOLOTracker(TrackerBase):  # - Inheritance, aggregation :)
    def __init__(self, model):
        self.model = model
        self.track_history = defaultdict(lambda: [])
        self.missing_frames = {}

    # Polymorphism :)
    def process_frame(self, frame):
        results = self.model.track(frame)
        return results

    def update_track_history(self, boxes, track_ids):
        current_ids = set(track_ids)

        # Add new detections to track history
        for box, track_id in zip(boxes, track_ids):
            x, y, _, _ = box
            self.track_history[track_id].append((float(x), float(y)))

        # Reduce track length even if object is missing
        for track_id in list(self.track_history.keys()):
            if track_id not in current_ids or len(self.track_history[track_id]) > 5:
                fade_speed = 1
                self.track_history[track_id] = self.track_history[track_id][fade_speed:]

            # If track becomes empty, remove it completely
            if not self.track_history[track_id]:
                del self.track_history[track_id]

    def draw_tracks(self, frame):
        for track_id, track in self.track_history.items():
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(0, 45, 255), thickness=1, lineType=cv2.LINE_AA)


class VideoHandler:
    def __init__(self, input_path):
        # CV2 INTERFACE CONFIG
        self.input_path = input_path
        self.cap = cv2.VideoCapture(input_path) # Capture

        output_path = input_path.split('.')[0] + '_tracked.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        frame_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        self.out = cv2.VideoWriter(output_path, fourcc, frame_rate, frame_size)
        # ----------------------

    # Encapsulation :)
    def read_frame(self):
        return self.cap.read()

    def write_frame(self, frame):
        self.out.write(frame)

    def release(self):
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()


class YOLOTrackingApp:
    def __init__(self, video_path):
        # Composition, setup
        self.video_handler = VideoHandler(video_path)
        self.tracker = YOLOTracker(YOLOModel())

    def run(self):
        while self.video_handler.cap.isOpened():
            success, frame = self.video_handler.read_frame() # cap returns 2 args
            if not success:
                break

            # Abstraction :)
            results = self.tracker.process_frame(frame)

            if results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu() # Tensoriai perkeliami į CPU nes CV2 klaunada
                track_ids = results[0].boxes.id.int().cpu().tolist() # CPU metodas .tolist()
                self.tracker.update_track_history(boxes, track_ids)

                annotated_frame = results[0].plot() # 1 frame procesingas
                self.tracker.draw_tracks(annotated_frame)

                self.video_handler.write_frame(annotated_frame)

        self.video_handler.release()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python yolov8_oop_tracker.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    app = YOLOTrackingApp(video_path)
    app.run()
