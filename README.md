# Clash of Clans - Gold Mine Detection & Heatmap

Real-time computer vision system to detect Gold Mines in Clash of Clans gameplay and overlay a Gaussian location heatmap using Ultralytics YOLO.

---

## 📁 Project Structure

```text
coc-automation/
├── dataset/
│   ├── raw/                 # Captured gameplay screenshots
│   ├── images/
│   │   ├── train/           # Training images
│   │   ├── val/             # Validation images
│   │   └── test/            # Test images
│   └── labels/
│       ├── train/           # YOLO .txt label files for train
│       ├── val/             # YOLO .txt label files for val
│       └── test/            # YOLO .txt label files for test
├── models/
│   └── gold_mine_best.pt    # Best trained model checkpoint
├── src/
│   ├── capture_screen.py    # Screenshot capture utility (press S to save)
│   ├── train.py             # YOLO fine-tuning script
│   ├── evaluate.py          # Model evaluation (mAP, Precision, Recall)
│   ├── detect_live.py       # Live detection & heatmap overlay
│   ├── detect_image.py      # Inference on static images
│   ├── heatmap.py           # Gaussian heatmap generator
│   ├── test_pipeline.py     # Self-test validation script
│   └── main.py              # CLI interactive runner
├── data.yaml                # Dataset configuration (class 0: gold_mine)
├── yolo11n.pt               # Pretrained YOLO11 nano base model
└── requirements.txt         # Project dependencies
```

---

## 🚀 Quick Start Guide

### 1. Collect Screenshots
Run the capture utility with your game or emulator open:
```powershell
.venv\Scripts\python src/capture_screen.py
```
- Press **`S`** to capture a frame (saved into `dataset/raw/`).
- Press **`Q`** to exit.

### 2. Annotate & Organize Dataset
Annotate the images using [Roboflow](https://roboflow.com), [CVAT](https://cvat.ai), or [LabelImg](https://github.com/HumanSignal/labelImg) with class `gold_mine` (class ID `0`), then export in YOLO format and place them into `dataset/images/` and `dataset/labels/`.

### 3. Train the Detector
```powershell
.venv\Scripts\python src/train.py --epochs 50 --batch 16
```
The best checkpoint is automatically copied to `models/gold_mine_best.pt`.

### 4. Evaluate the Model
```powershell
.venv\Scripts\python src/evaluate.py --split test
```

### 5. Run Real-Time Live Detection & Heatmap
```powershell
.venv\Scripts\python src/detect_live.py
```
**Controls during live detection:**
- `H`: Toggle Heatmap on/off
- `B`: Toggle Bounding Boxes on/off
- `+` / `-`: Adjust confidence threshold
- `S`: Save snapshot
- `Q`: Quit

