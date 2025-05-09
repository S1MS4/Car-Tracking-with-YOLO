import argparse
from Scripts.app import YOLOTrackingApp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO Object Tracking")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--model", default="yolov8m.pt", help="YOLO model to use")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="Confidence threshold for YOLOv8Model")
    args = parser.parse_args()

    app = YOLOTrackingApp(args.video_path, args.model, confidence_threshold=args.confidence_threshold)
    app.run()
