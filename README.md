# 🚗 Car Tracking with YOLO 🚀

## 📌 1. Introduction
### 🔍 What is your application?
This project is a **real-time car tracking system** using **YOLO (You Only Look Once)**. It detects and tracks vehicles in video streams, leveraging **Object-Oriented Programming (OOP) principles** for modular and scalable design.

### ▶️ How to run the program?
1. **Clone the repository:**
   ```sh
   git clone https://github.com/S1MS4/Car-Tracking-with-YOLO.git
   cd Car-Tracking-with-YOLO
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Download YOLO model weights** and place them in the appropriate directory.
4. **Run the car tracking system:**
   ```sh
   python yolo_video_main.py <video_path> --model yolov8m.pt
   ```

### 🎮 How to use the program?
- Load a video file by specifying the video path.
- The YOLO model detects and tracks vehicles in real-time.
- The results are displayed in an annotated video output and stored in a **tracked video file**.

---

## 📊 2. Body/Analysis
### 🛠️ How the program implements functional requirements
The `yolo_video_main.py` script is structured using **Object-Oriented Programming (OOP)** and includes the following components:

### 🔄 Core Classes and Responsibilities
#### **YOLOModel (Factory Pattern)** 🏭
- Loads a specified YOLO model.
- Provides a `track()` method to detect and track objects in a video frame.
- Implements a **Factory Pattern** (`YOLOModelFactory`) for flexible model selection.

#### **YOLOTracker (Tracking Logic)** 🎯
- Uses the YOLO model to process frames and detect objects.
- Maintains a **history** of object movements.
- Draws **bounding boxes** and **tracking lines** on the video.
- Uses **Encapsulation** to manage the internal tracking logic.

#### **VideoHandler (Video Processing)** 🎥
- Handles reading video frames and writing processed output.
- Uses **Abstraction** to provide a simplified video processing interface.
- Ensures proper file handling and cleanup.

#### **YOLOTrackingApp (Main Application)** 🚀
- Combines the **YOLO model, tracker, and video handler** into a structured pipeline.
- Uses a **command-line interface** for running the application.
- Implements **Encapsulation** by keeping model handling and tracking logic separate.

### 🔑 Object-Oriented Principles Used
- **Encapsulation**: Each class manages its own responsibilities.
- **Abstraction**: Provides simplified interaction with the tracking system.
- **Inheritance**: Allows modular extension for different models.
- **Polymorphism**: Supports different YOLO models dynamically.

### 🎨 Design Pattern Used
The **Factory Method Pattern** 🏭 is implemented to configure different YOLO models dynamically, making it easier to switch between versions (e.g., YOLOv8, YOLOv9, etc.).

### 📂 File Handling
The program saves detection results as a **tracked video file** (`<video_name>_tracked.mp4`), ensuring persistence of detected objects.

---

## 🏆 3. Results and Summary
### ✅ Results
- 🚗 Successfully detects and tracks vehicles in real-time.
- 📊 Outputs a processed video with bounding boxes and tracking lines.
- 🎯 Uses **OOP principles** for modular and scalable design.

### 🔍 Conclusions
- Demonstrates how **YOLO and OOP principles** can be used in real-world applications.
- Efficient and extendable structure for vehicle tracking.

### 🚀 Future Enhancements
- Multi-object tracking for various traffic elements.
- Improved tracking efficiency with optimized algorithms.
- Cloud integration for remote data processing.

---

## 📚 4. References
📖 [YOLO Documentation](https://pjreddie.com/darknet/yolo/)
📖 [Python unittest Documentation](https://docs.python.org/3/library/unittest.html)
📖 [PEP8 Style Guide](https://peps.python.org/pep-0008/)
