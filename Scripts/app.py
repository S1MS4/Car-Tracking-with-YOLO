from Scripts.video_handler import VideoHandler
from Scripts.yolo_model import YOLOModelFactory
from Scripts.yolo_tracker import YOLOTracker

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
                boxes = results[0].boxes.xywh.cpu().numpy()
                track_ids = results[0].boxes.id.int().tolist()
                self.tracker.update_history(boxes, track_ids)

                annotated_frame = results[0].plot()
                self.tracker.draw_tracks(annotated_frame)
                self.video.write_frame(annotated_frame)

        self.video.release()
