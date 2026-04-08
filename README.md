# 🐄 AI-Based Cow Detection System

**YOLOv8 · ByteTrack · OpenCows2020 · ROI Counting**

Trained on the [OpenCows2020 dataset](https://data.bris.ac.uk/data/dataset/10m32xl88x2b61zlkkgz3fml17) (University of Bristol) — 11,779 overhead images of Holstein-Friesian cattle. Detects, tracks, and counts cows crossing a virtual ROI line in real-time CCTV footage.

---

## What It Does

- **Detects** individual cows in overhead/top-down video frames using YOLOv8m
- **Tracks** each cow across frames with ByteTrack, assigning persistent IDs
- **Counts** cows crossing a configurable ROI line (IN / OUT direction-aware)
- **Monitors** zone occupancy — how many cows are inside a defined polygon area at any moment
- Outputs an annotated video, live HUD overlay, and a per-frame CSV log

---

## Demo

> Add your output video or GIF here — `results/demo.gif`

---

## Folder Structure

```
Ai-Based-Cow-Detection-System/
│
├── cows-detection-system.ipynb   # Full training pipeline (Kaggle notebook)
│                                 # Dataset download → preprocessing →
│                                 # YOLO conversion → YOLOv8 training
│
├── main.py                       # Inference script — detection + tracking + counting
│                                 # Run this locally on any video
│
├── GetCordinates.py              # Interactive tool to pick ROI line and polygon
│                                 # zone coordinates by clicking on a video frame
│
├── ExtractFrames.py              # Extracts frames from a video at a given interval
│                                 # Useful for building a custom dataset
│
├── getfrm.py                     # Quick frame sampler — grab N frames from a video
│
├── weights/
│   └── best.pt                   # Trained YOLOv8m weights (add after training)
│
└── input/
    └── your_video.mp4            # Place your inference video here
```

---

## Training Pipeline (Notebook)

`cows-detection-system.ipynb` covers the full pipeline on Kaggle GPU:

| Step | Detail |
|------|--------|
| Dataset download | OpenCows2020 via direct Dropbox URL — no `dataset-tools` required |
| Format | Supervisely JSON → YOLO normalised TXT |
| Split | 70% train / 20% val / 10% test |
| Model | YOLOv8m pretrained on COCO |
| Epochs | 50 + early stopping (patience=20) |
| Augmentation | Mosaic, MixUp, HSV, flip, scale |
| Input size | 640×640 |

### Why no `dataset-tools`?

`dataset-tools` pins `pandas<=1.5.2`, which has no pre-built wheel for Python 3.12 — the current Kaggle runtime. The notebook downloads directly from the Dropbox URL bundled inside the package's own JSON, skipping the broken install entirely.

---

## Local Inference

### 1. Install

```bash
pip install ultralytics supervision lapx opencv-python
```

### 2. Configure

Open `main.py` and set the `CONFIG` block at the top:

```python
CONFIG = {
    "weights" : "weights/best.pt",   # path to your trained weights
    "source"  : "input/video.mp4",   # your video file, or 0 for webcam

    # ROI counting line — coordinates in pixels
    # start = RIGHT end, end = LEFT end → "moving down" = IN (overhead camera)
    "line_start" : (1280, 360),
    "line_end"   : (0,    360),

    # Polygon zone — counts cows inside this area each frame
    "polygon_zone": {
        "enabled" : True,
        "points"  : np.array([(311,182),(511,182),(525,311),(315,314)], dtype=np.int64),
        "label"   : "Farm Area",
    },
}
```

### 3. Pick Your ROI Coordinates

Not sure what pixel coordinates to use? Run:

```bash
python GetCordinates.py
```

Click on the video frame to get exact pixel coordinates for your line and zone.

### 4. Run

```bash
# Process full video, save annotated output + CSV
python main.py

# With live preview window
python main.py --show

# Override paths from terminal
python main.py --weights weights/best.pt --source input/video.mp4 --show

# Webcam
python main.py --source 0 --show
```

### 5. Outputs

| File | Description |
|------|-------------|
| `cow_tracked_<name>.mp4` | Annotated video with boxes, trails, HUD, line |
| `cow_log_<name>.csv` | Per-frame log: tracker ID, bbox, conf, IN/OUT counts |
| Terminal | Final summary — unique cows, total IN/OUT, net change |

---

## HUD Display

Every output frame shows a live dashboard:

```
┌─────────────────────────────┐
│  COW TRACKER                │
├─────────────────────────────┤
│  Frame          :   1 240   │
│  Live FPS       :    28.3   │
│  Cows in frame  :       4   │
│  Unique IDs     :      11   │
├─────────────────────────────┤
│  Line ▶ IN      :       7   │
│  Line ◀ OUT     :       4   │
├─────────────────────────────┤
│  Zone (Farm Area):      2   │
└─────────────────────────────┘
```

Each cow also gets a colour-coded trail showing its movement path and a speed badge in px/s.

---

## Dataset

**OpenCows2020** — University of Bristol  
DOI: [10.5523/bris.10m32xl88x2b61zlkkgz3fml17](https://doi.org/10.5523/bris.10m32xl88x2b61zlkkgz3fml17)  
11,779 images · 13,026 labelled bounding boxes · single class: `cow`  
3 splits: `detection_and_localisation` / `identification-train` / `identification-test`

---

## Tech Stack

| Component | Library / Tool |
|-----------|---------------|
| Detection | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| Tracking | [ByteTrack](https://github.com/ifzhang/ByteTrack) via `supervision` |
| Annotation | [Supervision](https://github.com/roboflow/supervision) |
| Training | Kaggle GPU (T4 / P100) |
| Dataset | OpenCows2020 (University of Bristol) |

---

## Known Issues & Fixes Applied

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| Tracking not activating | `min_consec_frames=2` suppressed IDs for first 2 frames | Set to `1` |
| IN/OUT always zero | LineZone start/end direction swapped for overhead camera | Swapped endpoints |
| Video stops early | `cap.read()` returning `ret=False` on transient codec frames | Consecutive-failure counter (threshold=5) |
| Silent save failure | `cv2.VideoWriter` fails without raising an exception | Added `isOpened()` guard |

---

## Author

**Muhammad Qadeer**  
AI/ML Engineer · Computer Vision Research Assistant  
GIFT University, Pakistan  

[![GitHub](https://img.shields.io/badge/GitHub-QadeerDev-181717?logo=github)](https://github.com/QadeerDev)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-qadeerjutt-0A66C2?logo=linkedin)](https://linkedin.com/in/qadeerjutt)

---

## License

MIT License — see [LICENSE](LICENSE) for details.  
OpenCows2020 dataset is subject to University of Bristol terms — for academic use only.
