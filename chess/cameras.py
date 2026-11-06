"""Open cameras by device name instead of by OpenCV index (macOS reshuffles
indices whenever a camera is plugged in or reconnects).

OpenCV's macOS backend numbers cameras in AVFoundation uniqueID order, so the
name -> index map is rebuilt from that list (pyobjc). Resolution/fps cannot
tell the overhead cam apart: the ZEB, the leftover Iriun driver and the
MacBook camera all report 1920 wide.

    cap = open_camera(OVERHEAD)
    cap = open_camera(WRIST)
"""

import cv2

OVERHEAD = "overhead"
WRIST = "wrist"

FINGERPRINTS = {
    # ZEB at 640x480 is a 4:3 crop of its 16:9 sensor; the whole board still fits, and it is
    # what lerobot records at, so calibration matches the policy's view.
    OVERHEAD: dict(name="ZEB LIVE PRO", set_size=(640, 480)),
    WRIST: dict(name="USB2.0 PC CAMERA", set_size=(640, 480)),
}
MAX_INDEX = 6


def probe(idx):
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        return None
    props = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return props


def list_cameras():
    """[(index, width, default_fps)] for every openable index."""
    out = []
    for i in range(MAX_INDEX):
        p = probe(i)
        if p:
            out.append((i, int(p[0]), int(p[1])))
    return out


def device_names():
    """Camera names in OpenCV index order."""
    import AVFoundation as AV

    devs = AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo)
    return [str(d.localizedName()) for d in sorted(devs, key=lambda d: str(d.uniqueID()))]


def camera_index(kind):
    names = device_names()
    name = FINGERPRINTS[kind]["name"]
    if name not in names:
        raise RuntimeError(f"no {kind} camera ({name!r}) found; cameras in index order: {names}")
    return names.index(name)


def open_camera(kind, fps=30, mjpg=False):
    idx = camera_index(kind)
    cap = cv2.VideoCapture(idx)
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    w, h = FINGERPRINTS[kind]["set_size"]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError(f"{kind} camera (index {idx}) failed to open")
    print(f"{kind} camera: index {idx}")
    return cap


if __name__ == "__main__":
    names = device_names()
    for i, w, fps in list_cameras():
        print(f"index {i}: {names[i] if i < len(names) else '?'}, {w} wide, {fps} fps default")
    for kind in (OVERHEAD, WRIST):
        try:
            print(kind, "->", camera_index(kind))
        except RuntimeError as e:
            print(e)
