# ROS2 Camera Streaming + Remote YOLO Inference

This folder contains the ROS2 nodes for streaming camera frames from the Raspberry Pi on the TurtleBot to your computer, where a trained YOLO model runs inference and displays the annotated results.

## Architecture

```
┌─────────────────────┐     WiFi / LAN      ┌──────────────────────┐
│   Raspberry Pi      │ ─────────────────>  │   Your Computer      │
│   (on TurtleBot)    │  ROS2 CompressedImg │   (GPU / CUDA)       │
│                     │   /camera/image/... │                      │
│ pi_camera_publisher │                     │ computer_yolo_sub    │
│  - reads USB cam    │                     │  - subscribes        │
│  - JPEG-encodes     │                     │  - runs YOLOv8n      │
│  - publishes 15 FPS │                     │  - displays boxes    │
│                     │ <─────────────────  │  - (optional) pubs   │
│                     │  /yolo/annotated... │    back annotated    │
└─────────────────────┘                     └──────────────────────┘
```

## Files

- `pi_camera_publisher.py` — Runs on the **Pi**. Captures USB camera frames and publishes compressed JPEG frames over ROS2.
- `computer_yolo_subscriber.py` — Runs on the **computer**. Subscribes to the Pi's stream, runs YOLO inference, displays annotated feed.
- `computer_live_viewer.py` — Runs on the **computer**. Simple passthrough viewer (no YOLO) — use it to verify the stream works before adding inference.

---

## One-time setup

### 1. Install ROS2 on both the Pi and your computer

Use the same distro on both (Humble recommended for Ubuntu 22.04).

Pi: follow the ROS2 Humble install guide for ARM64.
Computer: follow the ROS2 Humble install guide for your OS.

### 2. Install Python dependencies

**On the Pi:**
```bash
sudo apt install python3-opencv
pip3 install numpy
```
(rclpy and sensor_msgs come with ROS2)

**On the computer:**
```bash
pip install ultralytics opencv-python numpy
```
(Plus CUDA + PyTorch for GPU acceleration — see the Ultralytics install guide)

### 3. Configure the network

Both machines need the **same ROS_DOMAIN_ID** to see each other. Add this to the `~/.bashrc` of BOTH the Pi and your computer:

```bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
```

Then `source ~/.bashrc`. They also need to be on the same network (same WiFi or same wired LAN).

### 4. Verify they can see each other

On both machines:
```bash
source /opt/ros/humble/setup.bash
ros2 topic list
```
Anything one publishes should be visible to the other.

---

## Running the system

### Test 1: Verify camera stream (no YOLO yet)

**Pi terminal:**
```bash
cd ~/ENEE408I/project/ros2_stream
python3 pi_camera_publisher.py
```

**Computer terminal:**
```bash
cd ~/ENEE408I/project/ros2_stream
python3 computer_live_viewer.py
```

You should see the Pi's camera feed on your computer. If yes, networking works. If no, check `ROS_DOMAIN_ID` and firewall.

### Test 2: Run YOLO inference on the stream

**Pi terminal (same as before):**
```bash
python3 pi_camera_publisher.py
```

**Computer terminal:**
```bash
python3 computer_yolo_subscriber.py --ros-args \
  -p model_path:=../camera1_topdown/runs/detect/camera1_topdown/weights/best.pt
```

You should see the same feed, now with YOLO bounding boxes drawn on detected packages. Press `q` to quit.

### Switching between cameras / models

For Camera 1 (top-down):
```bash
# Pi
python3 pi_camera_publisher.py --ros-args \
  -p camera_index:=0 \
  -p topic_name:=/camera1/image/compressed

# Computer
python3 computer_yolo_subscriber.py --ros-args \
  -p input_topic:=/camera1/image/compressed \
  -p model_path:=../camera1_topdown/runs/detect/camera1_topdown/weights/best.pt
```

For Camera 2 (angled):
```bash
# Pi
python3 pi_camera_publisher.py --ros-args \
  -p camera_index:=1 \
  -p topic_name:=/camera2/image/compressed

# Computer
python3 computer_yolo_subscriber.py --ros-args \
  -p input_topic:=/camera2/image/compressed \
  -p model_path:=../camera2_angled/runs/detect/camera2_angled/weights/best.pt
```

---

## Tuning for performance

If the stream is laggy:
- Lower `fps` on the publisher (e.g. 10 instead of 15)
- Lower `jpeg_quality` (e.g. 60 instead of 80) — saves bandwidth
- Lower `frame_width`/`frame_height` (e.g. 480x360)
- Make sure your computer has CUDA/GPU enabled; set `device:=0` on the subscriber

If detections are missing objects:
- Raise `conf_threshold` (e.g. 0.4) to reduce false positives, OR lower it (e.g. 0.15) to catch more
- Verify you're loading the correct trained model for the correct camera viewpoint

---

## Collecting training data through the ROS2 stream (optional)

Once the stream works, you could also capture training frames remotely:
just modify `computer_live_viewer.py` to save frames on keypress `p` (same logic
as `datacollector.py`). This lets you gather training images from the exact
camera + mount setup on the robot rather than tethered via USB.
