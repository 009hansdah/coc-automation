# Detailed Step-by-Step Guidelines

## Real-Time Clash of Clans Building Detection and Heatmap with CNN

## Project goal

Create an application that captures Clash of Clans gameplay frames,
detects selected buildings with a CNN-based object detector, and
overlays a heatmap showing the detected building locations.

Recommended starting approach: use a pretrained YOLO object detector and
fine-tune it on your own annotated images. YOLO is a CNN-based object
detection architecture. A simple image-classification CNN will not
provide bounding boxes or object locations by itself.

## Roadmap

1.  Set up Python and project folders.
2.  Collect screenshots.
3.  Define target classes.
4.  Annotate buildings.
5.  Organize and validate the dataset.
6.  Train the detector.
7.  Evaluate it on unseen base layouts.
8.  Capture the live gameplay window.
9.  Run real-time detection.
10. Generate heatmaps.
11. Combine everything into one application.
12. Improve speed and reliability.

------------------------------------------------------------------------

# Stage 1 --- Environment Setup

### 1.1 Create the project directory

``` text
coc_building_detection/
├── dataset/
│   ├── raw/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
├── models/
├── src/
│   ├── capture_screen.py
│   ├── train.py
│   ├── evaluate.py
│   ├── detect_live.py
│   ├── heatmap.py
│   └── main.py
├── data.yaml
├── requirements.txt
└── README.md
```

### 1.2 Create and activate a virtual environment

Windows PowerShell:

``` powershell
python -m venv venv
venv\Scripts\activate
```

Ubuntu/Linux:

``` bash
python -m venv venv
source venv/bin/activate
```

### 1.3 Install libraries

Create `requirements.txt`:

``` text
ultralytics
opencv-python
mss
numpy
matplotlib
```

Install:

``` bash
pip install -r requirements.txt
```

Confirm installation:

``` bash
yolo checks
```

For GPU training, install a PyTorch build compatible with your GPU and
CUDA environment using the official PyTorch installation selector.

------------------------------------------------------------------------

# Stage 2 --- Collect Gameplay Screenshots

### 2.1 Choose target building classes

Start with a manageable list:

    Class ID Class name
  ---------- -----------------
           0 town_hall
           1 inferno_tower
           2 eagle_artillery
           3 xbow
           4 air_defense

Use the same spelling and class order everywhere.

### 2.2 Collect diverse screenshots

Capture screenshots of a variety of bases and visual situations:

-   Different base layouts and Town Hall levels.
-   Buildings at different positions.
-   Different zoom levels, if available.
-   Partial occlusions, animations, and visual effects.
-   Different interface states, while prioritizing clear base views.

For a first experiment, try a few hundred varied screenshots. The number
of labeled building instances matters more than simply collecting many
near-duplicate frames. Ensure the dataset is representative of the game
views in which you intend to run inference.

### 2.3 Capture screenshots using MSS

Create `src/capture_screen.py`:

``` python
import cv2
import mss
import numpy as np
import os

os.makedirs("dataset/raw", exist_ok=True)

count = 0

with mss.mss() as sct:
    monitor = sct.monitors[1]

    while True:
        screenshot = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        cv2.imshow("Gameplay Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            path = f"dataset/raw/image_{count:05d}.jpg"
            cv2.imwrite(path, frame)
            print("Saved:", path)
            count += 1

        elif key == ord("q"):
            break

cv2.destroyAllWindows()
```

Run:

``` bash
python src/capture_screen.py
```

-   Press `S` to save a frame.
-   Press `Q` to quit.

This example captures the primary monitor. You can refine it to capture
only the emulator/game window or a fixed rectangle, which usually
improves performance and avoids labeling unrelated UI.

------------------------------------------------------------------------

# Stage 3 --- Annotate Images

### 3.1 Choose an annotation tool

Use one of: - Roboflow Annotate - CVAT - LabelImg (with YOLO format
support)

Create a project configured for object detection and enter the class
list from Stage 2.

### 3.2 Draw bounding boxes

For every target building in each screenshot:

1.  Draw a tight box around the visible building.
2.  Assign the correct class.
3.  Label every instance of the target classes.
4.  Do not label a building if its identity is genuinely unclear; follow
    a consistent policy for partial occlusions.
5.  Review labels for accidental class swaps and loose boxes.

Annotations must be consistent. Inconsistent boxes and incorrect class
labels can reduce detector quality substantially.

### 3.3 Export in YOLO format

YOLO labels are text files, one per image. Each labeled object is a
line:

``` text
<class_id> <x_center> <y_center> <width> <height>
```

All four coordinates are normalized to the image size, from 0 to 1.

Example:

``` text
0 0.512 0.438 0.120 0.155
1 0.270 0.620 0.085 0.110
```

These are illustrative values, not actual annotations. Export the
dataset in YOLO format from your annotation tool.

------------------------------------------------------------------------

# Stage 4 --- Split and Prepare the Dataset

### 4.1 Split images into train, validation, and test

A typical split:

-   Train: 70%
-   Validation: 20%
-   Test: 10%

Important: split by base layout or source session when possible.
Near-identical screenshots from one base should not be spread across
train and test, or test metrics may be overly optimistic.

### 4.2 Expected structure

``` text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Each image must have a matching label file with the same basename.
Images with no target objects may have empty label files if your dataset
format/tool expects them.

### 4.3 Create `data.yaml`

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

Check that `path`, folder names, and class indices agree with your
exported dataset.

### 4.4 Dataset quality checklist

-   Image files open correctly.
-   Each label filename matches an image filename.
-   Class IDs are within the range 0--4.
-   Bounding box values are normalized and valid.
-   Every target building is annotated according to your policy.
-   No train/test leakage from duplicate or near-duplicate base views.
-   Every class has examples in train and validation sets.

------------------------------------------------------------------------

# Stage 5 --- Train the CNN Object Detector

### 5.1 Select a pretrained model

Fine-tune a small pretrained YOLO model first. A nano/small model is a
practical starting point for experiments. The exact checkpoint name
depends on the Ultralytics version installed; check its current model
documentation.

### 5.2 Create `src/train.py`

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

### 5.3 Run training

``` bash
python src/train.py
```

Adjust: - `epochs`: number of passes over the training dataset. -
`imgsz`: input image size; higher sizes may help small buildings but use
more memory. - `batch`: reduce it if GPU memory is insufficient. -
`model`: choose a smaller model for limited hardware.

The best checkpoint is usually located in a training run folder similar
to:

``` text
runs/detect/coc_building_detector/weights/best.pt
```

Copy the best checkpoint into `models/` if you want a stable project
path.

### 5.4 Monitor training

Look at: - Training and validation losses. - Precision and recall. -
mAP50 and mAP50-95. - Per-class metrics. - False positives and missed
detections on validation images.

Do not choose a model based only on training loss. Validation
performance and qualitative predictions matter.

------------------------------------------------------------------------

# Stage 6 --- Evaluate the Model

Create `src/evaluate.py`:

``` python
from ultralytics import YOLO

model = YOLO("models/best.pt")

metrics = model.val(
    data="data.yaml",
    split="test",
    imgsz=640
)

print(metrics)
```

Test on layouts that were not used for training or tuning.

Review errors manually: - Is a Town Hall mistaken for another
building? - Are small buildings missed? - Are decorations or UI elements
falsely detected? - Does the detector work at the live game's resolution
and zoom?

If the model performs poorly, improve annotations and data diversity
before simply increasing epochs.

------------------------------------------------------------------------

# Stage 7 --- Capture Live Gameplay

### 7.1 Set up the game view

Open Clash of Clans in your chosen supported display/emulator
environment. Keep the game visible at a consistent resolution and zoom
during initial testing.

### 7.2 Capture frames

MSS can capture a monitor or rectangular region. For initial testing,
capture the primary monitor:

``` python
import mss
import numpy as np
import cv2

with mss.mss() as sct:
    monitor = sct.monitors[1]
    screenshot = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    cv2.imshow("Captured Frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

Before proceeding, verify that the displayed frame actually contains the
game and is not obscured by another window.

For a more efficient application, capture only the game window or the
rectangle containing the base. Window-specific capture can require
OS-specific libraries or configuration.

------------------------------------------------------------------------

# Stage 8 --- Run Real-Time Building Detection

Create `src/detect_live.py`:

``` python
import cv2
import mss
import numpy as np
from ultralytics import YOLO

model = YOLO("models/best.pt")

with mss.mss() as sct:
    monitor = sct.monitors[1]

    while True:
        screenshot = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        results = model.predict(
            frame,
            conf=0.50,
            imgsz=640,
            verbose=False
        )

        annotated = results[0].plot()

        cv2.imshow("Live Building Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
```

Run:

``` bash
python src/detect_live.py
```

Press `Q` to stop.

### Detection output

For each detection, the model provides: - Class ID. - Confidence. -
Bounding box: `x1, y1, x2, y2`.

These values will be used to create the heatmap.

------------------------------------------------------------------------

# Stage 9 --- Create the Heatmap

## 9.1 Understand the heatmap

A location heatmap assigns intensity around each detected building.
Overlapping spots become more intense.

This basic heatmap describes detected locations. It does not
automatically represent defense range, vulnerability, or attack
probability.

## 9.2 Create `src/heatmap.py`

``` python
import cv2
import numpy as np

def generate_heatmap(frame, centers, radius=50, sigma=25):
    height, width = frame.shape[:2]

    heat = np.zeros((height, width), dtype=np.float32)

    for x, y in centers:
        cv2.circle(
            heat,
            (int(x), int(y)),
            radius,
            1.0,
            -1
        )

    heat = cv2.GaussianBlur(
        heat,
        (0, 0),
        sigmaX=sigma
    )

    heat = cv2.normalize(
        heat,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    colored_heat = cv2.applyColorMap(
        heat,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        frame,
        0.65,
        colored_heat,
        0.35,
        0
    )

    return overlay
```

## 9.3 Extract building centers from detections

Use this snippet after running YOLO inference:

``` python
result = results[0]
centers = []

if result.boxes is not None:
    for box in result.boxes:
        class_id = int(box.cls[0])

        if class_id in [0, 1, 2, 3, 4]:
            coords = box.xyxy[0].cpu().numpy()

            x1, y1, x2, y2 = coords

            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            centers.append((x_center, y_center))
```

Then create the heatmap:

``` python
from heatmap import generate_heatmap

heatmap_frame = generate_heatmap(frame, centers)

cv2.imshow("Building Heatmap", heatmap_frame)
```

Integrate these lines into `detect_live.py`, replacing or adding to the
`results[0].plot()` display.

## 9.4 Per-building heatmaps

To make a heatmap for only one class, filter on `class_id`.

For example:

``` python
target_class = 1  # Inferno Tower

centers = []

for box in result.boxes:
    class_id = int(box.cls[0])

    if class_id == target_class:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
```

Repeat with other class IDs, or add a UI selector for the desired class.

------------------------------------------------------------------------

# Stage 10 --- Combine into the Main Application

The final application loop should follow this sequence:

``` text
Capture current gameplay frame
        ↓
Run object detector
        ↓
Read boxes, classes, confidence
        ↓
Filter detections by confidence and target class
        ↓
Compute building centers
        ↓
Generate heatmap
        ↓
Blend heatmap and original frame
        ↓
Display annotated frame
        ↓
Repeat until user quits
```

A useful final display can include: - Bounding boxes and labels. -
Confidence scores. - Heatmap overlay. - FPS display. - Class selection
or toggles. - A quit key.

------------------------------------------------------------------------

# Stage 11 --- Improve Performance and Reliability

### 11.1 Improve detection quality

-   Add examples for classes that are frequently missed.
-   Inspect and correct annotation mistakes.
-   Add more base-layout diversity.
-   Use a held-out test set.
-   Tune confidence threshold based on validation data.
-   Consider a larger input size for small targets, while measuring
    speed.

### 11.2 Improve FPS

-   Capture only the game region.
-   Use a nano/small detector.
-   Run inference on GPU if available.
-   Avoid unnecessary display and disk operations in the loop.
-   Measure actual FPS rather than assuming real-time performance.

### 11.3 Reduce heatmap flicker

A per-frame heatmap may flicker as detections change. Options include: -
Smooth heat values across successive frames. - Use object tracking to
associate detections between frames. - Use a short rolling accumulation
window. - Reset or decay heat values so stale detections do not remain
indefinitely.

### 11.4 Distinguish location heatmap from strategic analysis

A detection heatmap only shows where the detector finds buildings.

To create a strategic overlay such as defensive coverage or vulnerable
regions, you would need separate logic for: - Building ranges. -
Building type and level. - Map geometry. - Walls and pathing. - Intended
analysis goal.

Keep this as a later extension rather than mixing it into the first
detector.

------------------------------------------------------------------------

# Stage 12 --- Final Project Checklist

## Dataset

-   [ ] Target classes selected
-   [ ] Varied screenshots collected
-   [ ] Buildings annotated consistently
-   [ ] YOLO format exported
-   [ ] Train/validation/test split created
-   [ ] Dataset configuration checked

## Model

-   [ ] Pretrained detector fine-tuned
-   [ ] Validation metrics reviewed
-   [ ] Test images from unseen layouts evaluated
-   [ ] Best checkpoint saved

## Live inference

-   [ ] Gameplay frame capture works
-   [ ] Detector runs on live frames
-   [ ] Bounding boxes and labels display
-   [ ] Confidence filtering works

## Heatmap

-   [ ] Building centers extracted
-   [ ] Heatmap generated
-   [ ] Heatmap blended with live image
-   [ ] Per-class filtering tested
-   [ ] Flicker and performance reviewed

## Deliverable

-   [ ] Clean project folder
-   [ ] Requirements file
-   [ ] README with setup instructions
-   [ ] Example screenshot or demo video
-   [ ] Model metrics documented

## Recommended order of implementation

Do not try to build the entire application at once. Use these
milestones:

1.  Annotate 50--100 images and train a proof-of-concept model.
2.  Confirm it detects buildings in saved screenshots.
3.  Expand the dataset and improve the model.
4.  Add live screen capture.
5.  Add the heatmap.
6.  Optimize FPS and add class-specific visualization.

This staged approach makes debugging easier because you can isolate
dataset, model, capture, and visualization problems.
