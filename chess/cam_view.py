"""Live view of a camera with the measured frame rate in the corner.

Usage:  .venv/bin/python chess/cam_view.py wrist
        .venv/bin/python chess/cam_view.py top          (overhead)
        .venv/bin/python chess/cam_view.py [index]     (default 0; raw OpenCV index)
Keys:   s save a snapshot to chess/cam_snaps/   q quit
"""

import sys
import time
from pathlib import Path

import cv2

from cameras import OVERHEAD, WRIST, camera_index

SNAP_DIR = Path(__file__).parent / "cam_snaps"
NAMES = {"wrist": WRIST, "top": OVERHEAD, "overhead": OVERHEAD}


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "0"
    idx = camera_index(NAMES[arg]) if arg in NAMES else int(arg)
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        sys.exit(f"could not open camera {idx}")

    t0, n, fps = time.time(), 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        n += 1
        if time.time() - t0 >= 1.0:
            fps, n, t0 = n / (time.time() - t0), 0, time.time()
        h, w = frame.shape[:2]
        cv2.putText(frame, f"cam {idx}  {w}x{h}  {fps:.0f} fps", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(f"camera {idx}", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            SNAP_DIR.mkdir(exist_ok=True)
            path = SNAP_DIR / f"cam{idx}_{int(time.time())}.png"
            cv2.imwrite(str(path), frame)
            print("saved", path)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
