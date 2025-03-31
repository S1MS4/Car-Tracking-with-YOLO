import cv2
import numpy as np
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.pt")  # Load an official model

# Read an image
input_path = "test_image.jpg"
image = cv2.imread(input_path)

# Predict with the model
results = model(image)

# Use the plot method to draw bounding boxes
annotated_image = results[0].plot()

# Save the output image
output_path = input_path.rsplit('.', 1)[0] + '_tracked.jpg'
cv2.imwrite(output_path, annotated_image)

# Show the image
cv2.imshow("YOLO Detection", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
