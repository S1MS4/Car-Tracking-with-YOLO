import cv2
import torch
from ultralytics import YOLO

# Load the model
model_path = "yolo11m.pt"  # Ensure this file exists
model = YOLO(model_path)

# Load an image
image_path = "test_image.jpg"  # Replace with your image filename
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not read the image. Check the file path.")
    exit()

# Run object detection (NO tracking)
results = model.predict(image)

# Check if results contain boxes
if results[0].boxes:
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Get box coordinates (x1, y1, x2, y2)
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = results[0].boxes.cls.cpu().numpy()  # Class IDs
    names = results[0].names  # Class name mapping

    print("Detected objects (No IDs shown):")
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)  # Convert to integers
        confidence = confidences[i]  # Get confidence score
        class_id = int(class_ids[i])  # Get object class ID
        label = f"{names[class_id]}"  # Only class name (NO ID)

        # Customize box appearance
        color = (0, 255, 0)  # Green box
        thickness = 2  # Box thickness
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Draw class label above box
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        text_x, text_y = x1, y1 - 5  # Position above the box
        cv2.rectangle(image, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0], text_y), color, -1)  # Background for text
        cv2.putText(image, label, (text_x, text_y - 2), font, font_scale, (255, 255, 255), font_thickness)  # White text

    # Show the image with boxes (NO IDs)
    cv2.imshow("YOLO Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No objects detected.")
