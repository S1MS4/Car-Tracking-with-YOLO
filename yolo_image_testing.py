import cv2
from ultralytics import YOLO

model = YOLO("yolov8m.pt")

input_path = "test_image.jpg"
image = cv2.imread(input_path)

results = model(image)

annotated_image = results[0].plot()

output_path = input_path.rsplit('.', 1)[0] + '_tracked.jpg'
cv2.imwrite(output_path, annotated_image)

cv2.imshow("YOLO Detection", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
