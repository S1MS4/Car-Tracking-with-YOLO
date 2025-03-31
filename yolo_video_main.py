import cv2
import numpy as np
import argparse
from collections import defaultdict
from ultralytics import YOLO

# ---------------- Factory for YOLO Models ----------------
class YOLOModel:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def track(self, frame):
        return self.model.track(frame, persist=True)


class YOLOModelFactory:
    @staticmethod
    def create(model_path: str):
        return YOLOModel(model_path)

# ---------------- Tracker Class ----------------
class YOLOTracker:
    def __init__(self, model: YOLOModel):
        self.model = model
        self.history = defaultdict(list)

    def process_frame(self, frame):
        return self.model.track(frame)

    def update_history(self, boxes, track_ids):
        if not track_ids or not boxes.any():  # Prevent issues with empty lists
            return

        for track_id, (x, y, _, _) in zip(track_ids, boxes.tolist()):
            if track_id is None:
                continue  # Skip if track_id is missing

            if track_id not in self.history:
                self.history[track_id] = []  # Ensure key exists

            self.history[track_id].append((x, y))

        # Keep only the last 5 positions for each tracked object
        self.history = {k: v[-5:] for k, v in self.history.items() if k in track_ids}

    def draw_tracks(self, frame):
        for track_id, track in self.history.items():
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(0, 45, 255), thickness=1, lineType=cv2.LINE_AA)


# ---------------- Video Handling ----------------
class VideoHandler:
    def __init__(self, input_path: str):
        self.cap = cv2.VideoCapture(input_path)
        self.out = self._init_writer(input_path)

    def _init_writer(self, input_path: str):
        output_path = input_path.rsplit('.', 1)[0] + '_tracked.mp4'
        fourcc = cv2.VideoWriter.fourcc(*'mp4v') # codec
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    def read_frame(self):
        return self.cap.read()

    def write_frame(self, frame):
        self.out.write(frame)

    def release(self):
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()


# ---------------- Main Application ----------------
class YOLOTrackingApp:
    def __init__(self, video_path: str, model_path: str):
        self.video = VideoHandler(video_path)
        model = YOLOModelFactory.create(model_path)
        self.tracker = YOLOTracker(model)

    def run(self):
        while self.video.cap.isOpened():
            success, frame = self.video.read_frame()
            if not success:
                break

            results = self.tracker.process_frame(frame)
            if results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu().numpy() # cpu method transforms received gpu tensors
                track_ids = results[0].boxes.id.int().tolist() # compatible cpu format
                self.tracker.update_history(boxes, track_ids)

                annotated_frame = results[0].plot()
                self.tracker.draw_tracks(annotated_frame)
                self.video.write_frame(annotated_frame)

        self.video.release()


# ---------------- Command-Line Interface ----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO Object Tracking")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--model", default="yolov8m.pt", choices=["yolov8m.pt","yolov9m.pt","yolov10m.pt","yolo11m.pt","yolo12m.pt"], help="YOLO model to use")

    args = parser.parse_args()

    app = YOLOTrackingApp(args.video_path, args.model)
    app.run()