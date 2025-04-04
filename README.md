# 🚗 Car Tracking with YOLO >>UNFINISHED<<🎯

Welcome to the **Car Tracking with YOLO** project This nifty tool lets you detect and track vehicles in videos using the powerful YOLO (You Only Look Once) deep learning mode. Whether you're analyzing traffic patterns or just curious about vehicle movements, this project has got you covere!

## 🌟 Features

- **Real-time Vehicle Detection*: Spot cars in your videos as they moe.
- **Easy Integration*: Seamlessly plug into your existing video processing pipelins.
- **Modular Design*: Customize components to fit your unique nees.

## 🚀 Getting Started

Follow these simple steps to set up and run the project:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/S1MS4/Car-Tracking-with-YOLO.git
   ```


2. **Navigate to the Project Directory**:

   ```bash
   cd Car-Tracking-with-YOLO
   ```


3. **Install Dependencies**:

   Ensure you have Python installed. Then, install the required packages:

   ```bash
   pip install -r requirements.txt
   ```


4. **Run the Application**:

   Execute the main script to start processing:

   ```bash
   python Scripts/run.py --input path_to_video.mp4
   ```


   Replace `path_to_video.mp4` with the path to your video file.

## 🛠️ Under the Hood

### Object-Oriented Principles in Action

Our codebase is crafted with solid OOP principles:

- **Encapsulatio**: The `VideoHandler` class wraps all video processing details, keeping things idy.

  
```python
  class VideoHandler:
      def __init__(self, input_path: str):
          self.cap = cv2.VideoCapture(input_path)
          self.out = self._init_writer(input_path)
      # ...
 ```


- **Abstractio**: The `YOLOModel` class hides the complexities of the YOLO model, offering a clean interace.

  
```python
  class YOLOModel:
      def __init__(self, model_path: str):
          self.model = YOLO(model_path)
      # ...
 ```


- **Inheritanc**: While not heavily used now, the design is ready for future classes to inherit and extend functionaliies.

- **Polymorphis**: The `YOLOModelFactory` class showcases polymorphism by standardizing the creation of YOLO model instaces.

  
```python
  class YOLOModelFactory:
      @staticmethod
      def create(model_path: str):
          return YOLOModel(model_path)
 ```


### Design Patterns

We've incorporated the **Factory Method** pattern with our `YOLOModelFactoy`. This approach streamlines the creation of `YOLOModel` instances, making the code more flexible and scalble.

### Composition and Aggregation

- **Compositio**: The `YOLOTracker` class contains an instance of `YOLOModel`, enabling efficient frame procesing.

  
```python
  class YOLOTracker:
      def __init__(self, model):
          self.model = model
      # ...
 ```


- **Aggregatio**: The `VideoHandler` class manages OpenCV resources, handling video capture and writing seamlesly.

### File Operatins

Our `VideoHandler` class takes care of reading from and writing to video files, ensuring your processed videos are saved without a htch.

### Testng

Quality is key! We've used the `unittest` framework to test core functionalities. Check out `yolo_unit_test.py` for tests on model creation, frame processing, and ore


```python
import unittest
import numpy as np
from Scripts.app import YOLOModelFactory, YOLOTracker, VideoHandler

class TestYOLOIntegration(unittest.TestCase):
    def setUp(self):
        self.model = YOLOModelFactory.create("yolo11m.pt")
        self.tracker = YOLOTracker(self.model)
    # ...
```


### Code Stle

We adhere to PEP8 style guidelines, ensuring our code is clean and readble.

## 🎯 Achievements & Challenges

- **Successs**: Integrated the YOLO model for real-time vehicle tracking, achieving impressive detection seeds.

- **Hurdls**: Encountered challenges with detection accuracy in low-light videos, prompting us to enhance preprocessing technques.

## 🌈 Future Directions

- **Enhanced Accurcy**: Implement advanced techniques to boost detection in varied lighting condtions.

- **User-Friendly Interfce**: Develop a GUI to make the application more accessible to non-technicalusers.

- **Expanded Object Trackng**: Extend capabilities to track multiple object types, broadening the application'sscope.

## 🔗 References

- [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)
- [OpenCV Documentation](https://docs.opencv.org/

---

*Dive in, explore, and happy trcking!* 
