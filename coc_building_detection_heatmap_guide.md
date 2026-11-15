# Real-Time Clash of Clans Building Detection and Heatmap

## 1. Project Overview

Build a computer-vision system that captures live Clash of Clans
gameplay, detects selected buildings, and draws a heatmap over the game
image to show where those buildings are located.

The system will:

1.  Capture frames from the game window or emulator.
2.  Preprocess the frames for the detector.
3.  Run a trained object-detection model.
4.  Identify building classes and their bounding boxes.
5.  Convert detections into spatial points or regions.
6.  Generate and overlay a heatmap.
7.  Display the annotated live frame.

> Recommended approach: use a YOLO object-detection model (such as
> Ultralytics YOLO) rather than a basic image-classification CNN.
> Classification predicts a label for an entire image; object detection
> predicts labels and locations for individual objects.

## 2. Architecture

``` mermaid
flowchart TD
    A[Clash of Clans gameplay] --> B[Screen capture]
    B --> C[Frame preprocessing]
    C --> D[YOLO CNN detector]
    D --> E[Building class, box, confidence]
    E --> F[Filter detections]
    F --> G[Create heatmap]
    G --> H[Blend heatmap with frame]
    H --> I[Live display]
```

## 3. Suggested Technology Stack

  Component                 Suggested tools
  ------------------------- -------------------------------
  Language                  Python
  Screen capture            MSS
  Image processing          OpenCV
  Detection model           Ultralytics YOLO
  Deep learning framework   PyTorch (used by Ultralytics)
  Dataset annotation        Roboflow or CVAT
  Heatmap                   NumPy and OpenCV
  Optional evaluation       Precision, recall, mAP

## 4. Choose Building Classes

Start with a small set of classes. For example:

-   `town_hall`
-   `inferno_tower`
-   `eagle_artillery`
-   `xbow`
-   `air_defense`

You can add more classes once the initial model works. Use class names
consistently in annotations and the dataset configuration.

## 5. Build the Dataset

### 5.1 Capture screenshots

Collect screenshots from a range of bases and situations:

-   Different Town Hall levels and layouts.
-   Different zoom levels, if applicable.
-   Buildings partly obscured by effects or overlays.
-   Different lighting, animation, and UI situations.
-   Different base designs, not just repeated screenshots of one base.

For an initial experiment, aim for a few hundred varied screenshots.
More diverse, correctly annotated data is generally more useful than
many near-duplicate frames.

### 5.2 Annotate images

Use Roboflow, CVAT, or another YOLO-compatible annotation tool.

For each target building: - Draw a tight bounding box around the
building. - Assign the correct class. - Avoid labeling objects you
cannot confidently identify. - Keep annotation rules consistent.

### 5.3 Split the dataset

A reasonable split is:

-   70% training
-   20% validation
-   10% test

Split by base/layout where possible. Avoid placing nearly identical
frames from the same base into both training and test sets, as this can
make evaluation misleading.

### 5.4 YOLO dataset structure

``` text
coc-building-detection/
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
├── data.yaml
├── train.py
├── detect_live.py
└── heatmap.py
```

Each image has a matching `.txt` label file. YOLO labels use this
format:

``` text
<class_id> <x_center> <y_center> <width> <height>
```

Coordinates are normalized to values from 0 to 1, relative to image
width and height.

### 5.5 Create `data.yaml`

``` yaml
path: dataset
train: images/train
val: images/val
test: images/test

names:
  0: town_hall
  1: inferno_tower
  2: eagle_artillery
  3: xbow
  4: air_defense
```

Ensure the class IDs match the annotation tool's class ordering.

## 6. Install Dependencies

Create a virtual environment, then install the packages:

``` bash
python -m venv .venv
```

Activate it on Windows:

``` powershell
.venv\Scripts\activate
```

Activate it on Linux:

``` bash
source .venv/bin/activate
```

Install packages:

``` bash
pip install ultralytics opencv-python mss numpy
```

For GPU training, install a PyTorch build compatible with your CUDA
setup using the official PyTorch installation selector.

## 7. Train the Model

Create `train.py`:

``` python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="coc_building_detector"
)
```

Run training:

``` bash
python train.py
```

Notes: - `yolo11n.pt` is a small pretrained model suitable for initial
experiments. Confirm the model version supported by your installed
Ultralytics package. - Start with a smaller batch size if you run out of
GPU memory. - Inspect validation results and misdetections; do not rely
only on training loss. - The best checkpoint is commonly saved under
`runs/detect/coc_building_detector/weights/best.pt`.

## 8. Capture the Live Game Screen

The game can be shown through an emulator or a windowed game. MSS can
capture a selected screen region. Example below captures the primary
monitor; adjust the monitor or region for your setup.

Create `detect_live.py`:

``` python
import cv2
import mss
import numpy as np
from ultralytics import YOLO

model = YOLO("runs/detect/coc_building_detector/weights/best.pt")

with mss.mss() as sct:
    monitor = sct.monitors[1]

    while True:
        screenshot = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        results = model.predict(
            frame,
            conf=0.5,
            imgsz=640,
            verbose=False
        )

        annotated = results[0].plot()

        cv2.imshow("CoC Building Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
```

Press `q` to quit. For better performance, capture only the game window
or its region instead of the entire monitor. The monitor capture example
is a starting point; window-specific capture may require additional
setup.

## 9. Generate a Building Detection Heatmap

A simple detection heatmap places a Gaussian-like spot at the center of
every detected target building. This represents detected building
locations, not defensive range or attack probability.

Create `heatmap.py`:

``` python
import cv2
import numpy as np

def generate_heatmap(frame, detections, radius=50, sigma=25):
    height, width = frame.shape[:2]
    heat = np.zeros((height, width), dtype=np.float32)

    for x, y in detections:
        cv2.circle(
            heat,
            (int(x), int(y)),
            radius,
            1.0,
            -1
        )

    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma)
    heat = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX)
    heat = heat.astype(np.uint8)

    colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.65, colored, 0.35, 0)

    return overlay
```

### Use detections from YOLO

After prediction, get the centers of the detected boxes and pass them
into the heatmap function:

``` python
result = results[0]
centers = []

if result.boxes is not None:
    for box in result.boxes:
        class_id = int(box.cls[0])

        # Include only selected target classes.
        if class_id in [0, 1, 2, 3, 4]:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            centers.append((x_center, y_center))

heatmap_frame = generate_heatmap(frame, centers)
cv2.imshow("Building Heatmap", heatmap_frame)
```

Integrate this into the live detection loop, replacing or supplementing
`results[0].plot()`.

## 10. Separate Heatmaps by Building Type

To create a separate heatmap for each class, filter detections by class
ID before calling `generate_heatmap`.

For example: - Class 0: Town Hall heatmap. - Class 1: Inferno Tower
heatmap. - Class 2: Eagle Artillery heatmap.

This makes the visualization more informative than combining all
buildings into one layer.

## 11. Improve the Project

Possible enhancements:

-   Show class labels and confidence scores.
-   Add a confidence threshold slider.
-   Capture only the game window for higher FPS.
-   Use tracking to reduce flickering between frames.
-   Add per-class heatmap toggles.
-   Map screen positions to a stable base-grid coordinate system.
-   Estimate defensive coverage using building range data (a different
    task from detection density).
-   Save screenshots and detection logs for evaluation.
-   Measure FPS, precision, recall, and mAP on held-out layouts.

## 12. Common Issues

  -----------------------------------------------------------------------
  Problem                             Possible fix
  ----------------------------------- -----------------------------------
  Building is missed                  Add varied examples, check
                                      annotation quality, train longer,
                                      or adjust confidence threshold

  Wrong building class                Improve class-specific data and
                                      annotation consistency

  Heatmap is too bright               Reduce circle intensity, radius, or
                                      overlay weight

  Heatmap is blurry                   Reduce Gaussian sigma

  Low FPS                             Capture a smaller region, use a
                                      smaller model, or use GPU inference

  Model works on training bases but   Split data by base layout and
  not new bases                       collect more diverse examples

  UI elements get detected as         Include negative examples and crop
  buildings                           to the gameplay area
  -----------------------------------------------------------------------

## 13. Development Milestones

1.  Collect and annotate screenshots.
2.  Train a detector for 3--5 building classes.
3.  Evaluate it on unseen base layouts.
4.  Run inference on captured game frames.
5.  Generate and overlay a location heatmap.
6.  Add per-class heatmaps and performance optimizations.

## Expected Result

A live gameplay window with detected building bounding boxes and a
heatmap overlay that highlights the locations of selected Clash of Clans
buildings.

**Important:** A location heatmap only visualizes detections. To produce
a strategic heatmap such as defensive coverage, weak spots, or attack
routes, you will need additional game-specific geometry and logic.
