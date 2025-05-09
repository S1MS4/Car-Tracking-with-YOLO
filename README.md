---

# 🎯 YOLO Object Tracking Application

## 📌 Introduction

### 🔍 What is this application?

This is a **YOLO-based object tracking system** that processes a video file, detects and tracks multiple objects using the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) model, and outputs an annotated video with tracked object paths. It draws persistent tracks for each object using a combination of YOLO’s object detection and custom tracking history.
## 👁️‍🗨️ Preview

The program processes each frame of your video, assigns consistent IDs to objects, and draws motion trails. These trails are shortened intentionally to keep the visualization clean and focused on active detections.

![Demo GIF](show.gif)

Above: Objects are detected, labeled, and tracked with persistent paths as they move through the scene. (Note there are prediction inaccuracies)
### ▶️ How to run the program

Make sure you have the required dependencies installed. They are listed below:
---

## 📦 Dependencies

The application requires the following Python libraries:

| Component                  | Purpose                                                             |
|----------------------------|---------------------------------------------------------------------|
| `cv2` (OpenCV)             | Reading, writing, and drawing on video frames                      |
| `numpy`                    | Numerical operations and array manipulation                        |
| `argparse`                 | Parses command-line arguments                                      |
| `collections.defaultdict`  | Efficiently tracks object history                                  |
| `ultralytics`              | YOLO model for object detection and tracking                      |
| `unittest`                 | Built-in module for writing and running tests                     |
| **`video`**               | 🔑 **Required**: The video file to be processed (input required for tracking) |
| **`Scripts`**             | 🔑 **Required**: Contains the main application logic and entry point |

⚠️ Note: The YOLO models (e.g., yolov8m.pt) are downloaded automatically by the ultralytics library if not already present locally. No manual download is required.
### 🔧 Installation

You can install the required dependencies using pip like this:

```bash
pip install opencv-python numpy ultralytics
```
**(NOTICE) before you run this don't forget to cd into the appropriate directory!**

**for me it's:**
```bash
cd yolov8_tracking/Car-Tracking-with-yolov8  
```
Run the program with:

```bash
python -m Scripts.run <path_to_input_video> --model yolov8m.pt
```

For example:

```bash
python -m Scripts.run "path/to/input/video" --model yolov8m.pt
```

### 🛠 How to use the program

1. Provide a path to a video file.
2. The application will process the video and generate an output file with `_model_name` appended to the filename.
3. You’ll find the output video in the same directory as your input.

---

## 🧠 Body / Analysis

### 📌 Object-Oriented Programming Principles

This project demonstrates all **four pillars of OOP**:

---

#### 1. **Encapsulation**

Encapsulation involves bundling data with the methods that operate on that data, while restricting direct access to some of the object's components. This principle is demonstrated throughout the project:

- The `VideoHandler` class encapsulates all video I/O logic. Users interact with simple public methods like `read_frame()` and `write_frame()`, without needing to understand how video capture or writing is initialized.

- A key example of encapsulation is the `_init_writer()` method, which is used internally by `VideoHandler` to configure the output video writer. It abstracts the complexity of determining output filename, format, frame size, and encoding:

```python
def _init_writer(self, input_path: str):
    output_path = input_path.rsplit('.', 1)[0] + '_tracked.mp4'
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    fps = self.cap.get(cv2.CAP_PROP_FPS)
    frame_size = (
        int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)
````

* This method is prefixed with an underscore to indicate it's intended for internal use only, reinforcing encapsulation by convention.

* Similarly, classes like `YOLOModel` and `YOLOTracker` expose only the methods needed by the main application, while hiding lower-level details like inference parameters or track history management.
---

#### 2. **Abstraction**

Abstraction is achieved by exposing only essential methods and hiding the internal complexities of how tasks are performed.

* A clear example of abstraction in this project is the introduction of the `BaseModel` abstract class using Python's `abc` module. This class defines a contract (`track()`) that all derived model classes must implement, allowing the rest of the application to interact with any model uniformly without needing to understand the specifics of each implementation:

```python
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, model_path: str):
        self.model_path = model_path

    @abstractmethod
    def track(self, frame):
        pass
```

* Two concrete implementations derive from `BaseModel`:

  * `YOLOModel`: A general wrapper for the Ultralytics YOLO model.
  * `YOLOv8Model`: A version with optional confidence threshold filtering for more control over detection quality.

```python
class YOLOModel(BaseModel):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.model = YOLO(model_path)

    def track(self, frame):
        return self.model.track(frame, persist=True, verbose=False)
```

```python
class YOLOv8Model(BaseModel):
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        super().__init__(model_path)
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def track(self, frame):
        results = self.model.track(frame, persist=True, verbose=False)
        if self.confidence_threshold > 0:
            results = self._filter_by_confidence(results)
        return results

    def _filter_by_confidence(self, results):
        for result in results:
            boxes = result.boxes
            confidences = boxes.conf
            mask = confidences >= self.confidence_threshold
            result.boxes = boxes[mask]
        return results
```

* The `YOLOModelFactory` further abstracts model instantiation by selecting the correct model class based on the filename or configuration:

```python
class YOLOModelFactory:
    @staticmethod
    def create(model_path: str, confidence_threshold: float = 0.5) -> BaseModel:
        if "v8" in model_path.lower():
            return YOLOv8Model(model_path, confidence_threshold)
        else:
            return YOLOModel(model_path)
```

* This abstraction ensures that the `YOLOTrackingApp` does not need to change even if new model types are introduced, as long as they conform to the `BaseModel` interface.
---
#### 3. **Inheritance**

Inheritance allows a class to reuse logic and structure from another class, reducing redundancy and increasing consistency.

This project demonstrates inheritance in multiple areas:

#### ✅ Custom Model Abstraction

* The `BaseModel` class uses inheritance to define a common interface for all YOLO models via the `track()` method. Both `YOLOModel` and `YOLOv8Model` inherit from it:

```python
class YOLOModel(BaseModel):
    ...
```

```python
class YOLOv8Model(BaseModel):
    ...
```

* This use of inheritance enforces a standard method structure and allows easy swapping or extension of model behavior.

#### ✅ Unit Testing

* The `TestYOLOIntegration` class inherits from Python’s built-in `unittest.TestCase`, gaining access to rich testing tools like `assertEqual()`, `setUp()`, etc.:

```python
class TestYOLOIntegration(unittest.TestCase):
    ...
```

---
#### 4. **Polymorphism**

Polymorphism is used via the **Factory Design Pattern** and potential extensibility:

* The `YOLOModelFactory` provides a unified interface for model creation, enabling different model types or configurations to be loaded without changing how they're used:

```python
class YOLOModelFactory:
    @staticmethod
    def create(model_path: str):
        return YOLOModel(model_path)
```

* Regardless of which YOLO model version is loaded (`yolov8m.pt`, `yolov8n.pt`, etc.), the calling code in `YOLOTrackingApp` does not change — it simply calls `model.track(frame)`.

---

### 🧩 Composition and Aggregation

The project demonstrates both **composition** and **aggregation** principles:

* **Composition** is used in `YOLOTrackingApp`, where the components `VideoHandler`, `YOLOModel`, and `YOLOTracker` are created and managed internally:

```python
class YOLOTrackingApp:
    def __init__(self, video_path: str, model_path: str):
        self.video = VideoHandler(video_path)
        model = YOLOModelFactory.create(model_path)
        self.tracker = YOLOTracker(model)
```

* These components have no meaning outside of the application—they are fully owned and controlled by it.

* **Aggregation** is used in `YOLOTracker` to maintain tracking history using a `defaultdict(list)`. This allows objects (track histories) to exist independently and be updated over time without tight coupling.

---

### 🧾 File I/O Operations

The project performs both **reading from** and **writing to** files:

* **Input** is handled using OpenCV’s `VideoCapture` to read frames from a video file.
* **Output** is written frame-by-frame to a new video file with `_tracked.mp4` suffix, using `VideoWriter`.

```python
self.cap = cv2.VideoCapture(input_path)
...
self.out.write(frame)
```

This ensures real-time processing of input videos and persistent saving of results.

---

### 🧪 Testing

Testing is implemented with Python's built-in `unittest` framework:

* Tests in `yolo_unit_test.py` verify:

  * Model creation (`test_model_creation`)
  * Frame processing logic (`test_process_frame`)
  * Tracker history updates (`test_update_history`)
  * Video file I/O (`test_video_handler`)

Example test:

```python
def test_update_history(self):
    boxes = np.array([[100, 100, 50, 50]])
    track_ids = [1]
    self.tracker.update_history(boxes, track_ids)
    self.assertIn(1, self.tracker.history)
```

This ensures components work individually and can be confidently composed into a larger system.

## 📊 Results and Summary

### ✅ Results

* 🧠 Successfully tracked objects frame-to-frame with persistent ID tracking.
* 🎨 Drew motion trails for each object to visualize movement.
* 💾 Saved annotated videos with object bounding boxes and tracks.
* 🧪 All unit tests passed successfully.

---

## 🧾 Conclusions

* 📌 This project demonstrates a practical implementation of object tracking using modern deep learning models and classical computer vision.
* 🧱 The structure is modular and testable, with proper abstraction between components.
* 🔧 Future work can introduce support for real-time video streams and multiple model types.
* 💡 Potential for extension into surveillance, traffic monitoring, or sports analytics applications.
* 🚀 YOLO models (especially YOLOv8) offer an impressive balance of speed and accuracy, making them well-suited for real-time applications even on modest hardware configurations.

* 📉 Despite strong performance, YOLO occasionally produces false positives or misses small/overlapping objects, suggesting room for improvement via post-processing techniques or hybrid tracking systems.
---

## 🚀 Extensibility Ideas

* Add GUI for video input/output selection.
* Live webcam tracking.
* Store tracking logs (CSV/JSON).
* Add model benchmarking and FPS measurement utilities.

---

## 📚 Resources & References

* [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
* [OpenCV Documentation](https://docs.opencv.org/)
* [Python Unittest Library](https://docs.python.org/3/library/unittest.html)
* [OOP Principles](https://en.wikipedia.org/wiki/Object-oriented_programming)

---
