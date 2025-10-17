import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import cv2
import numpy as np
from handLandmarks.handLandmarksDetection import (
    MediaPipeTaskHandLandmarkDetector,
)


# Hard-coded root folder for the hand dataset
DATASET_ROOT = Path(
    r"C:\Users\Staff\OneDrive - University of Greenwich\HandsDatasets\handRGBD\eval"
)
CSV_FILE = DATASET_ROOT / "reference_table.csv"
IMAGES_DIR = DATASET_ROOT / "rgb"

class HandBoundingBoxNotFound(RuntimeError):
    """Raised when no hand bounding box can be inferred for an image."""


_HAND_DETECTOR: Optional[MediaPipeTaskHandLandmarkDetector] = None


def _get_hand_detector() -> MediaPipeTaskHandLandmarkDetector:
    """Lazily instantiate the MediaPipe hand landmark detector."""
    global _HAND_DETECTOR
    if _HAND_DETECTOR is None:
        _HAND_DETECTOR = MediaPipeTaskHandLandmarkDetector()
    return _HAND_DETECTOR


def _detect_hand_bbox(image_bgr: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Run the hand landmark detector and return a bounding box in pixel coords.

    Returns (xmin, ymin, xmax, ymax). Raises HandBoundingBoxNotFound if missing.
    """
    detector = _get_hand_detector()
    _, landmarks_px = detector.detect(image_bgr)

    if landmarks_px.size == 0:
        raise HandBoundingBoxNotFound("Hand landmarks not detected.")

    img_h, img_w = image_bgr.shape[:2]
    xs = np.clip(landmarks_px[:, 0], 0, max(img_w - 1, 0))
    ys = np.clip(landmarks_px[:, 1], 0, max(img_h - 1, 0))

    xmin = int(xs.min())
    xmax = int(xs.max())
    ymin = int(ys.min())
    ymax = int(ys.max())
    return xmin, ymin, xmax, ymax


def load_dataset(update_csv: bool = True) -> List[Tuple[Dict[str, str], Path, np.ndarray]]:
    """
    Load metadata and RGB images listed in reference_table.csv.

    When update_csv is True, the function writes the detected bounding boxes to
    the `bbox` column in reference_table.csv (creating it if needed).
    """
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")

    if not IMAGES_DIR.is_dir():
        raise FileNotFoundError(f"Image directory not found: {IMAGES_DIR}")

    samples: List[Tuple[Dict[str, str], Path, np.ndarray]] = []
    updated_rows: List[Dict[str, str]] = []

    with CSV_FILE.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or "name" not in reader.fieldnames:
            raise ValueError("CSV file must contain a 'name' column.")

        fieldnames = list(reader.fieldnames)
        if "bbox" not in fieldnames:
            fieldnames.append("bbox")

        for row in reader:
            image_path = IMAGES_DIR / f"{row['name']}.png"
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

            if image_bgr is None:
                raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

            bbox_value = row.get("bbox", "").strip()
            if not bbox_value:
                try:
                    bbox = _detect_hand_bbox(image_bgr)
                except HandBoundingBoxNotFound as exc:
                    warnings.warn(f"{exc} for {image_path}", RuntimeWarning)
                    row["bbox"] = ""
                except Exception as exc:  # pragma: no cover - bubble up with context
                    warnings.warn(
                        f"Hand bbox detection failed for {image_path}: {exc}",
                        RuntimeWarning,
                    )
                    row["bbox"] = ""
                else:
                    row["bbox"] = ",".join(str(coord) for coord in bbox)
            else:
                row["bbox"] = bbox_value
            updated_rows.append(row)

            # Convert BGR (OpenCV default) to RGB for downstream use
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            samples.append((row.copy(), image_path, image_rgb))

    if update_csv:
        with CSV_FILE.open("w", newline="", encoding="utf-8") as csv_out:
            writer = csv.DictWriter(csv_out, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)

    return samples


if __name__ == "__main__":
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} samples from {IMAGES_DIR}")
    print(f"Updated bounding boxes stored in {CSV_FILE}")
