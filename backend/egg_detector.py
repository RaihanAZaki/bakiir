
import threading

import cv2
import numpy as np
from ultralytics import YOLOWorld

DETECTOR_MODEL = "yolov8s-world.pt"
DETECTION_THRESHOLD = 0.20

_detector = YOLOWorld(DETECTOR_MODEL)
_detector.set_classes(["egg"])
_detector_lock = threading.Lock()


def detect_egg(rgb: np.ndarray) -> dict:
    height, width = rgb.shape[:2]

    bgr = cv2.cvtColor(
        rgb, cv2.COLOR_RGB2BGR
    )

    with _detector_lock:
        results = _detector.predict(
            source=bgr,
            conf=DETECTION_THRESHOLD,
            imgsz=640,
            device="cpu",
            verbose=False,
        )

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return {
            "detected": False,
            "status": "not_detected",
            "confidence": None,
            "box": None,
            "image_width": width,
            "image_height": height,
        }

    best_index = int(
        boxes.conf.argmax().item()
    )

    best = boxes[best_index]

    x1, y1, x2, y2 = (
        best.xyxy[0].cpu().numpy().tolist()
    )

    x1 = max(0, min(width, int(x1)))
    y1 = max(0, min(height, int(y1)))
    x2 = max(0, min(width, int(x2)))
    y2 = max(0, min(height, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return {
            "detected": False,
            "status": "invalid_box",
            "confidence": None,
            "box": None,
            "image_width": width,
            "image_height": height,
        }

    return {
        "detected": True,
        "status": "egg_candidate_detected",
        "confidence": float(
            best.conf[0].item()
        ),
        "box": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        },
        "image_width": width,
        "image_height": height,
    }


def crop_detected_egg(
    rgb: np.ndarray,
    box: dict,
    padding_ratio: float = 0.12,
) -> np.ndarray:
    height, width = rgb.shape[:2]

    x1, y1 = box["x1"], box["y1"]
    x2, y2 = box["x2"], box["y2"]

    padding = int(
        max(x2 - x1, y2 - y1)
        * padding_ratio
    )

    left = max(0, x1 - padding)
    top = max(0, y1 - padding)
    right = min(width, x2 + padding)
    bottom = min(height, y2 + padding)

    return rgb[top:bottom, left:right].copy()
