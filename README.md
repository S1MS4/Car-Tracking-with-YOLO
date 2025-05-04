---

# 🎯 YOLO Object Tracking Application

## 📌 Introduction

### 🔍 What is this application?

This is a **YOLO-based object tracking system** that processes a video file, detects and tracks multiple objects using the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) model, and outputs an annotated video with tracked object paths. It draws persistent tracks for each object using a combination of YOLO’s object detection and custom tracking history.

### ▶️ How to run the program

Make sure you have the required dependencies installed.

Run the program with:

```bash
python -m Scripts.run <path_to_input_video> --model yolov8m.pt
```

**before you run this don't forget to cd into:**

```bash
cd yolov8_tracking/Car-Tracking-with-yolov8  
```

For example:

```bash
python -m Scripts.run "path/to/input/video" --model yolov8m.pt
```

### 🛠 How to use the program

1. Provide a path to a video file.
2. The application will process the video and generate an output file with `_tracked` appended to the filename.
3. You’ll find the output video in the same directory as your input.

---

## 🧠 Body / Analysis

### 📌 Object-Oriented Programming Principles

This project demonstrates all **four pillars of OOP**:

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


#### 2. **Abstraction**

Users only need to call `YOLOTrackingApp.run()` to process a video; internal details like frame handling, model inference, and drawing are hidden.

```python
class YOLOTrackingApp:
    def run(self):
        ...
```

#### 3. **Inheritance**

While no explicit inheritance hierarchy is shown, the use of YOLO from `ultralytics` implies inheritance behind the scenes. You could easily extend `YOLOModel` to create specialized variants.

#### 4. **Polymorphism**

Used through the factory design pattern—allowing for flexible creation of different models using the same interface.

```python
class YOLOModelFactory:
    @staticmethod
    def create(model_path: str):
        return YOLOModel(model_path)
```

### 🎯 Design Pattern Used

**Factory Pattern** is implemented in `YOLOModelFactory`, which abstracts away the instantiation of the YOLO model and allows flexible integration of future model types.

### 🔗 Composition and Aggregation

* **Composition**: `YOLOTrackingApp` *has-a* `VideoHandler`, `YOLOModel`, and `YOLOTracker`.
* **Aggregation**: `YOLOTracker` aggregates history data using Python’s `defaultdict`.

```python
class YOLOTrackingApp:
    def __init__(...):
        self.video = VideoHandler(...)
        ...
```

### 📂 File I/O (Read & Write)

* **Reading frames** from video via `cv2.VideoCapture`
* **Writing frames** with annotations via `cv2.VideoWriter`

```python
def read_frame(self):
    return self.cap.read()

def write_frame(self, frame):
    self.out.write(frame)
```

### 🧪 Testing

Unit testing is implemented via Python’s `unittest` module in `yolo_unit_test.py`.

Sample tests include:

* Model creation
* Frame processing
* Tracker update logic
* Video file loading

```python
def test_update_history(self):
    boxes = np.array([[100, 100, 50, 50]])
    track_ids = [1]
    self.tracker.update_history(boxes, track_ids)
    self.assertIn(1, self.tracker.history)
```

---

## 📊 Results and Summary

### ✅ Results

* 🧠 Successfully tracked objects frame-to-frame with persistent ID tracking.
* 🎨 Drew motion trails for each object to visualize movement.
* 💾 Saved annotated videos with object bounding boxes and tracks.
* 🧪 All unit tests passed successfully.
* 🔍 Minor challenge in managing frame rate and resolution consistency across different video files.

---

## 🧾 Conclusions

* 📌 This project demonstrates a practical implementation of object tracking using modern deep learning models and classical computer vision.
* 🧱 The structure is modular and testable, with proper abstraction between components.
* 🔧 Future work can introduce support for real-time video streams and multiple model types.
* 💡 Potential for extension into surveillance, traffic monitoring, or sports analytics applications.

---

## 🚀 Extensibility Ideas

* Add GUI for video input/output selection.
* Live webcam tracking.
* Store tracking logs (CSV/JSON).
* Multi-camera synchronization and fusion.
* Add model benchmarking and FPS measurement utilities.

---

## 📚 Resources & References

* [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
* [OpenCV Documentation](https://docs.opencv.org/)
* [Python Unittest Library](https://docs.python.org/3/library/unittest.html)
* [OOP Principles](https://en.wikipedia.org/wiki/Object-oriented_programming)

---
