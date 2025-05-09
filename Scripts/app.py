from tqdm import tqdm
from Scripts.video_handler import VideoHandler,cv2
from Scripts.yolo_model import YOLOModelFactory
from Scripts.yolo_tracker import YOLOTracker

class YOLOTrackingApp:
    def __init__(self, video_path: str, model_path: str, confidence_threshold: float = 0.5):
        self.video = VideoHandler(video_path,model_path)
        model = YOLOModelFactory.create(model_path, confidence_threshold)
        self.tracker = YOLOTracker(model)

    def run(self):
        total_frames = int(self.video.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = tqdm(total=total_frames, desc="Processing frames", unit="frame")

        while self.video.cap.isOpened():
            success, frame = self.video.read_frame()
            if not success:
                break

            results = self.tracker.process_frame(frame)
            if results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()
                track_ids = results[0].boxes.id.int().tolist()
                self.tracker.update_history(boxes, track_ids)

                annotated_frame = results[0].plot(line_width=1, font_size=1)
                self.tracker.draw_tracks(annotated_frame)
                self.video.write_frame(annotated_frame)

            progress_bar.update(1)
        progress_bar.close()

        self.video.release()
