# ENEE408I Project — TurtleBot Conveyor Belt Vision System

Two independent YOLO training pipelines, one per camera.

## Folder structure

```
project/
├── camera1_topdown/     ← USBFHD01M (overhead view of conveyor)
└── camera2_angled/      ← Second camera (45° side view for stacking detection)
```

Each camera folder contains the same self-contained pipeline:
- `datacollector.py` — live camera feed, press `p` to capture frames
- `datalabeler.py` — draw bounding boxes on captured frames
- `datasplit.py` — split labeled data into train/valid/test (70/20/10)
- `datatrainer.py` — train, test, or run live inference with YOLOv8n
- `data.yaml` — class configuration
- `data_raw/` — captured frames
- `data_labeled/` — YOLO-format `.txt` labels
- `train/` `valid/` `test/` — split dataset for YOLO

## Workflow (per camera)

```bash
cd project/camera1_topdown    # or camera2_angled

python datacollector.py       # capture frames (press 'p')
python datalabeler.py         # draw boxes (click x2, then 'n')
python datasplit.py           # split into train/valid/test
python datatrainer.py         # mode=0 to train, mode=1 to test, mode=2 for live
```

## Setting the right CAMERA_INDEX

If only ONE camera is plugged in → set `CAMERA_INDEX = 0` in both scripts.

If BOTH are plugged in simultaneously → use `CAMERA_INDEX = 0` for the first
and `CAMERA_INDEX = 1` for the second. The index assignment is OS-dependent,
so you may need to swap them.

Quick check: run this to see which indices work:
```python
import cv2
for i in range(4):
    cam = cv2.VideoCapture(i)
    print(f"Index {i}: {'OK' if cam.isOpened() else 'NO'}")
    cam.release()
```
