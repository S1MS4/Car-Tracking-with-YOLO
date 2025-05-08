import cv2
import os

class VideoHandler:
    def __init__(self, input_path: str, model_path: str):
        self.cap = cv2.VideoCapture(input_path)
        self.out = self._init_writer(input_path, model_path)

    def _init_writer(self, input_path: str, model_path: str):
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        output_path = input_path.rsplit('.', 1)[0] + f'_{model_name}.mp4'
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_size = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        return cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    def read_frame(self):
        return self.cap.read()

    def write_frame(self, frame):
        self.out.write(frame)

    def release(self):
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
