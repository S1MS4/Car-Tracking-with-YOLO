import cv2
import numpy as np
from ultralytics import YOLO

# Load model and image
model = YOLO("yolov8m.pt")
cv2.namedWindow("Traffic Light Detection", cv2.WINDOW_NORMAL)
image = cv2.imread("green_light.png")  # Replace with your image path

# Run inference
results = model(image)[0]

for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
    class_id = int(cls.item())
    class_name = results.names[class_id]

    if "traffic light" in class_name.lower():
        x1, y1, x2, y2 = map(int, box)

        # Crop traffic light region
        light_roi = image[y1:y2, x1:x2]

        # Convert to HSV
        hsv = cv2.cvtColor(light_roi, cv2.COLOR_BGR2HSV)

        # Define color thresholds
        red_lower1 = np.array([0, 70, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 70, 50])
        red_upper2 = np.array([180, 255, 255])

        green_lower = np.array([40, 50, 50])
        green_upper = np.array([90, 255, 255])

        # Create masks
        red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)

        red_pixels = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(green_mask)

        # Decide the light color and set border color accordingly
        color_label = "Unknown"
        border_color = (0, 255, 0)  # Default border to green

        if red_pixels > green_pixels and red_pixels > 50:
            color_label = "Red Light"
            border_color = (0, 0, 255)  # Red border for red light
        elif green_pixels > red_pixels and green_pixels > 50:
            color_label = "Green Light"
            border_color = (0, 255, 0)  # Green border for green light

        # Draw result on image with dynamic border color
        cv2.rectangle(image, (x1, y1), (x2, y2), border_color, 2)
        cv2.putText(image, color_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, border_color, 2)

# Show and save the result
cv2.namedWindow("Traffic Light Detection", cv2.WINDOW_NORMAL)
cv2.imshow("Traffic Light Detection", image)
cv2.imwrite("test_image_tracked.jpg", image)  # Save the result to a new file
cv2.waitKey(0)
cv2.destroyAllWindows()
