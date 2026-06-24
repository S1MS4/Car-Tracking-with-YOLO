# YOLO Object Tracking Application

## Introduction

This project was developed as part of an Object-Oriented Programming course using Python. It implements a YOLO-based object tracking system that processes a video file, detects and tracks multiple objects using the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) model, and produces an annotated output video with tracked object paths. The system draws persistent motion trails for each detected object by combining YOLO's detection capabilities with custom tracking history management.

## Preview

The program processes each frame of a video, assigns consistent IDs to objects, and draws motion trails. The trails are intentionally limited in length to keep the visualization focused on active detections.

![Demo GIF](show.gif)

Objects are detected, labeled, and tracked with persistent paths as they move through the scene. Note that prediction inaccuracies may occur.

### How to Run

Ensure that all required dependencies listed below are installed before running the application.

---

## Dependencies

The application requires the following Python libraries:

| Component                  | Purpose                                                             |
|----------------------------|---------------------------------------------------------------------|
| `cv2` (OpenCV)             | Reading, writing, and drawing on video frames                      |
| `numpy`                    | Numerical operations and array manipulation                        |
| `argparse`                 | Parses command-line arguments                                      |
| `collections.defaultdict`  | Efficiently tracks object history                                  |
| `ultralytics`              | YOLO model for object detection and tracking                       |
| `unittest`                 | Built-in module for writing and running tests                      |
| **`video`**                | Required: The video file to be processed                           |
| **`Scripts`**              | Required: Contains the main application logic and entry point      |

Note: YOLO model weights (e.g., `yolov8m.pt`) are downloaded automatically by the `ultralytics` library if not already present locally. No manual download is required.

### Installation

Install the required dependencies using pip:

```bash
pip install opencv-python numpy ultralytics
```

Before running the program, navigate to the appropriate directory:

```bash
cd yolov8_tracking/Car-Tracking-with-yolov8
```

Run the program with:

```bash
python -m Scripts.run <path_to_input_video> --model yolov8m.pt
```

Example:

```bash
python -m Scripts.run "path/to/input/video" --model yolov8m.pt
```

### Usage

1. Provide a path to a video file as a command-line argument.
2. The application processes the video and generates an output file with `_tracked` appended to the filename.
3. The output video is saved to the same directory as the input file.

---

## Body / Analysis

### Object-Oriented Programming Principles

This project demonstrates all four pillars of OOP.

---

#### 1. Encapsulation

Encapsulation involves bundling data with the methods that operate on it, while restricting direct access to internal components. This principle appears throughout the project:

- The `VideoHandler` class encapsulates all video I/O logic. Users interact with simple public methods like `read_frame()` and `write_frame()`, without needing to understand how video capture or writing is initialized.

- A key example is the `_init_writer()` method, which is used internally by `VideoHandler` to configure the output video writer. It abstracts the complexity of determining the output filename, format, frame size, and encoding:

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
```

- The underscore prefix indicates this method is intended for internal use only, reinforcing encapsulation by convention.

- Similarly, classes such as `YOLOModel` and `YOLOTracker` expose only the methods required by the main application, while hiding lower-level details such as inference parameters or track history management.

---

#### 2. Abstraction

Abstraction is achieved by exposing only essential methods and hiding the internal complexity of how tasks are performed.

- The `BaseModel` abstract class, defined using Python's `abc` module, establishes a contract (`track()`) that all derived model classes must implement. This allows the rest of the application to interact with any model uniformly, without needing to understand the specifics of each implementation:

```python
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, model_path: str):
        self.model_path = model_path

    @abstractmethod
    def track(self, frame):
        pass
```

- Two concrete implementations derive from `BaseModel`:

  - `YOLOModel`: A general wrapper for the Ultralytics YOLO model.
  - `YOLOv8Model`: A version with optional confidence threshold filtering for greater control over detection quality.

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

- The `YOLOModelFactory` further abstracts model instantiation by selecting the correct model class based on the filename or configuration:

```python
class YOLOModelFactory:
    @staticmethod
    def create(model_path: str, confidence_threshold: float = 0.5) -> BaseModel:
        if "v8" in model_path.lower():
            return YOLOv8Model(model_path, confidence_threshold)
        else:
            return YOLOModel(model_path)
```

- This design ensures that `YOLOTrackingApp` does not need to change when new model types are introduced, as long as they conform to the `BaseModel` interface.

---

#### 3. Inheritance

Inheritance allows a class to reuse logic and structure from a parent class, reducing redundancy and improving consistency.

**Custom Model Abstraction**

- The `BaseModel` class defines a common interface for all YOLO models via the `track()` method. Both `YOLOModel` and `YOLOv8Model` inherit from it:

```python
class YOLOModel(BaseModel):
    ...
```

```python
class YOLOv8Model(BaseModel):
    ...
```

This enforces a standard method structure and allows model implementations to be swapped or extended with minimal changes.

**Unit Testing**

- The `TestYOLOIntegration` class inherits from Python's built-in `unittest.TestCase`, gaining access to testing utilities such as `assertEqual()` and `setUp()`:

```python
class TestYOLOIntegration(unittest.TestCase):
    ...
```

---

#### 4. Polymorphism

Polymorphism is applied via the Factory Design Pattern:

- The `YOLOModelFactory` provides a unified interface for model creation, allowing different model types to be loaded without changing how they are used:

```python
class YOLOModelFactory:
    @staticmethod
    def create(model_path: str):
        return YOLOModel(model_path)
```

- Regardless of which YOLO model version is loaded (`yolov8m.pt`, `yolov8n.pt`, etc.), the calling code in `YOLOTrackingApp` does not change. It simply calls `model.track(frame)`.

---

### Composition and Aggregation

The project demonstrates both composition and aggregation:

- **Composition** is used in `YOLOTrackingApp`, where the components `VideoHandler`, `YOLOModel`, and `YOLOTracker` are created and managed internally:

```python
class YOLOTrackingApp:
    def __init__(self, video_path: str, model_path: str):
        self.video = VideoHandler(video_path)
        model = YOLOModelFactory.create(model_path)
        self.tracker = YOLOTracker(model)
```

These components have no meaningful existence outside of the application. They are fully owned and controlled by it.

- **Aggregation** is used in `YOLOTracker` to maintain tracking history using a `defaultdict(list)`. This allows track history objects to exist and be updated over time without tight coupling to the tracker itself.

---

### File I/O Operations

The project performs both reading from and writing to files:

- **Input** is handled using OpenCV's `VideoCapture` to read frames from a video file.
- **Output** is written frame-by-frame to a new video file with a `_tracked.mp4` suffix, using `VideoWriter`.

```python
self.cap = cv2.VideoCapture(input_path)
...
self.out.write(frame)
```

This ensures real-time processing of input video and persistent saving of results.

---

### Testing

Testing is implemented with Python's built-in `unittest` framework. Tests in `yolo_unit_test.py` verify:

- Model creation (`test_model_creation`)
- Frame processing logic (`test_process_frame`)
- Tracker history updates (`test_update_history`)
- Video file I/O (`test_video_handler`)

Example test:

```python
def test_update_history(self):
    boxes = np.array([[100, 100, 50, 50]])
    track_ids = [1]
    self.tracker.update_history(boxes, track_ids)
    self.assertIn(1, self.tracker.history)
```

This ensures that individual components work correctly and can be confidently composed into a larger system.

---

## Results and Summary

### Results

- Objects were successfully tracked frame-to-frame with persistent ID assignment.
- Motion trails were drawn for each object to visualize movement over time.
- Annotated output videos with bounding boxes and tracks were saved correctly.
- All unit tests passed successfully.

---

## Conclusions

- This project demonstrates a practical implementation of object tracking using modern deep learning models and classical computer vision techniques.
- The codebase is modular and testable, with clear abstraction boundaries between components.
- Future work could introduce support for real-time video streams and additional model types.
- Potential applications include surveillance, traffic monitoring, and sports analytics.
- YOLOv8 offers a strong balance of speed and accuracy, making it well-suited for real-time use even on modest hardware.
- Despite strong overall performance, the model occasionally produces false positives or misses small and overlapping objects, suggesting room for improvement through post-processing techniques or hybrid tracking approaches.

---

## Extensibility Ideas

- Add a GUI for video input and output selection.
- Support live webcam tracking.
- Export tracking logs to CSV or JSON format.
- Add model benchmarking and FPS measurement utilities.

---

## Resources and References

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Python Unittest Library](https://docs.python.org/3/library/unittest.html)
- [OOP Principles](https://en.wikipedia.org/wiki/Object-oriented_programming)
- [OpenAI](https://chatgpt.com/)
- [StackOverflow](https://stackoverflow.com/questions)
