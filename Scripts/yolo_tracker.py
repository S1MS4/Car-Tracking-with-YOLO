import numpy as np
import cv2
from collections import defaultdict

class YOLOTracker:
    def __init__(self, model):
        self.model = model
        self.history = defaultdict(list)

    def process_frame(self, frame):
        return self.model.track(frame)

    def update_history(self, boxes, track_ids):
        if not track_ids or not boxes.any():
            return
        for track_id, (x, y, _, _) in zip(track_ids, boxes.tolist()):
            if track_id is None:
                continue
            self.history[track_id].append((x, y))
        new_history = defaultdict(list)
        for k, v in self.history.items():
            if k in track_ids:
                new_history[k] = v[-5:]

        self.history = new_history

    def draw_tracks(self, frame):
        for track_id, track in self.history.items():
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(0, 45, 255), thickness=1, lineType=cv2.LINE_AA)
