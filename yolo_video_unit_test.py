import unittest
import numpy as np
from yolo_video_main import YOLOModelFactory, YOLOTracker, VideoHandler

class TestYOLOIntegration(unittest.TestCase):
    def setUp(self):
        self.model = YOLOModelFactory.create("yolo11m.pt")
        self.tracker = YOLOTracker(self.model)

    def test_model_creation(self):
        self.assertIsNotNone(self.model)

    def test_process_frame(self):
        frame = np.zeros((640, 480, 3), dtype=np.uint8)
        results = self.tracker.process_frame(frame)
        self.assertIsNotNone(results)

    def test_update_history(self):
        boxes = np.array([[100, 100, 50, 50]])
        track_ids = [1]
        self.tracker.update_history(boxes, track_ids)
        self.assertIn(1, self.tracker.history)

    def test_video_handler(self):
        handler = VideoHandler("test_video.mp4")
        self.assertTrue(handler.cap.isOpened())
        handler.release()

if __name__ == "__main__":
    unittest.main()
