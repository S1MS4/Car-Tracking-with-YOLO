import argparse
from Scripts.app import YOLOTrackingApp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO Object Tracking")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--model", default="yolov8m.pt", help="YOLO model to use")
    args = parser.parse_args()

    app = YOLOTrackingApp(args.video_path, args.model)
    app.run()
