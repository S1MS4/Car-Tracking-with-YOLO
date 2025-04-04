---

# YOLO Object Tracking Script 🎯📹

## Overview 🌟
This script provides an object tracking solution based on the YOLO (You Only Look Once) deep learning model. The system takes a video as input, applies YOLO object detection, tracks detected objects across frames, and outputs the video with the tracked objects annotated. The script leverages the `ultralytics` YOLO implementation for detection and utilizes OpenCV for video processing and visualization.

---

## Functionality ⚙️

### 1. **YOLOModel Class** 🧑‍💻
   - **Purpose**: Encapsulates the YOLO model for object detection and tracking.
   - **Constructor (`__init__`)**: Initializes the YOLO model from a specified file path (`model_path`).
   - **Method (`track`)**: Performs object tracking on a given video frame and returns the results (including bounding boxes and object IDs).

### 2. **YOLOModelFactory Class** 🏭
   - **Purpose**: Provides a factory method to create and return an instance of the `YOLOModel`.
   - **Static Method (`create`)**: Creates and returns a `YOLOModel` instance by loading a YOLO model from the specified path.

### 3. **YOLOTracker Class** 🏃‍♂️
   - **Purpose**: Manages the tracking history of objects detected across frames.
   - **Constructor (`__init__`)**: Initializes the tracker with the provided YOLO model.
   - **Method (`process_frame`)**: Passes a frame to the YOLO model for object detection and tracking.
   - **Method (`update_history`)**: Updates the tracking history with the current positions (bounding box) and IDs of the detected objects.
   - **Method (`draw_tracks`)**: Draws the tracking path for each object across multiple frames, visually representing the movement of each object.

### 4. **VideoHandler Class** 🎬
   - **Purpose**: Handles video I/O, including reading frames from an input video and writing the processed frames (with object tracking annotations) to an output video.
   - **Constructor (`__init__`)**: Initializes the video reader and output writer.
   - **Method (`_init_writer`)**: Initializes the video writer with proper codec, frame rate, and frame size.
   - **Method (`read_frame`)**: Reads the next frame from the video.
   - **Method (`write_frame`)**: Writes a processed frame (with tracked objects) to the output video.
   - **Method (`release`)**: Releases the video reader and writer, closing the video file.

### 5. **YOLOTrackingApp Class** 🚀
   - **Purpose**: Orchestrates the entire tracking process, from reading the video frames to tracking objects and writing the annotated video output.
   - **Constructor (`__init__`)**: Initializes the video handler, creates the YOLO model through the factory, and initializes the YOLO tracker.
   - **Method (`run`)**: Main loop to process the video frame by frame. For each frame:
     1. The frame is passed to the tracker for object detection.
     2. The tracking history is updated.
     3. Tracking paths are drawn on the frame.
     4. The processed frame is written to the output video.

### 6. **Command-Line Interface (CLI)** 🖥️
   - **Purpose**: Allows the user to run the script from the command line.
   - **Arguments**:
     - `video_path` (required): Path to the input video file to be processed.
     - `--model` (optional): Specifies the YOLO model to use. Available options are:
       - `yolov8m.pt`
       - `yolov9m.pt`
       - `yolov10m.pt`
       - `yolo11m.pt`
       - `yolo12m.pt`
     - The default model is `yolov8m.pt`.
   - **Usage**:
     ```bash
     python yolo_tracking.py path/to/video.mp4 --model yolov8m.pt
     ```

---

## Supported Color Modes 🎨
The `YOLOTracker` class uses a custom color scheme to visualize the tracking paths. You can customize the drawing of tracking paths to use different colors and thicknesses for visual clarity. 

Here are the supported colors:
- **Tracked Object Path Color**: 
  - Default: `RGB(0, 45, 255)` (Orange color).
  - You can modify this color in the `draw_tracks` method of the `YOLOTracker` class to suit your needs.
  
Additionally, the video itself is saved in **MP4 format** with the **'mp4v' codec** to maintain high compatibility with most video players.

---

## How It Works 🛠️

1. **Input**: The user provides a video file and specifies the YOLO model to use.
2. **Detection and Tracking**:
   - The video frames are processed one by one.
   - YOLO is used to detect objects, and the tracker keeps track of each object's ID and position across frames.
   - The object’s historical positions are drawn onto the video as polylines representing the movement over time.
3. **Output**: The annotated video, showing the tracked objects, is saved with the suffix `_tracked` appended to the original video filename.

---

## Dependencies 📦
- `opencv-python` (cv2): For video handling and frame manipulation.
- `numpy`: For handling arrays and numerical operations.
- `argparse`: For command-line argument parsing.
- `ultralytics`: For the YOLO model and object detection.

---

## Example Usage 🎥

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository/yolo-tracking.git
   ```

2. Install dependencies:
   ```bash
   pip install opencv-python numpy ultralytics
   ```

3. Run the script with an example video and a specified YOLO model:
   ```bash
   python yolo_tracking.py input_video.mp4 --model yolov8m.pt
   ```

4. The output video with tracked objects will be saved as `input_video_tracked.mp4`.

---

## Notes ⚠️
- Ensure that the specified model file (`yolov8m.pt`, etc.) is available in the correct directory or provide the full path to the model.
- The script saves the output video in the same directory as the input video with the suffix `_tracked`.

---

## Conclusion 🏁
This script offers a straightforward method to apply YOLO-based object tracking to video files, providing real-time tracking visualizations with minimal setup. It is flexible and can be adapted to work with different YOLO model versions, making it suitable for various detection and tracking tasks.
