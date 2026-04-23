"""
Pi Camera Sender
=================
Run this on the Raspberry Pi.
Captures from two cameras and streams to laptop.

Usage:
  python3 pi_sender.py
"""

import cv2
import socket
import struct
import pickle

# ── Edit these ──────────────────────────────────────────
CAM1         = 0     # top-down camera index
CAM2         = 2     # side angled camera index
PORT         = 9999
JPEG_QUALITY = 80    # 0-100, lower = faster but worse quality
# ────────────────────────────────────────────────────────

cap1 = cv2.VideoCapture(f"/dev/video{CAM1}")
cap2 = cv2.VideoCapture(f"/dev/video{CAM2}")

for cap in [cap1, cap2]:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)

if not cap1.isOpened():
    raise RuntimeError(f"Cannot open camera {CAM1}")
if not cap2.isOpened():
    raise RuntimeError(f"Cannot open camera {CAM2}")

print(f"Both cameras opened. Waiting for laptop connection on port {PORT}...")

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('0.0.0.0', PORT))
server.listen(1)
conn, addr = server.accept()
print(f"Laptop connected from {addr}")

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

try:
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("Failed to read from camera")
            continue

        _, enc1 = cv2.imencode('.jpg', frame1, encode_param)
        _, enc2 = cv2.imencode('.jpg', frame2, encode_param)

        payload = pickle.dumps((enc1, enc2))
        size    = struct.pack('>L', len(payload))
        conn.sendall(size + payload)

except (BrokenPipeError, ConnectionResetError):
    print("Laptop disconnected")
except KeyboardInterrupt:
    print("\nShutting down...")
finally:
    cap1.release()
    cap2.release()
    conn.close()
    server.close()
