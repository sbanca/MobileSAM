import cv2
import numpy as np
from pathlib import Path
from typing import Iterable


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest white component and drop smaller blobs."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


def fill_black_islands(mask: np.ndarray, max_hole_area: int = 2000) -> np.ndarray:
    """Fill small black holes inside the main white region."""
    inv_mask = cv2.bitwise_not(mask)
    num_labels_inv, labels_inv, stats_inv, _ = cv2.connectedComponentsWithStats(inv_mask, connectivity=8)
    filled_mask = mask.copy()

    for i in range(1, num_labels_inv):
        x, y, w, h, area = stats_inv[i]
        touches_border = (
            x <= 0 or y <= 0 or (x + w) >= inv_mask.shape[1] or (y + h) >= inv_mask.shape[0]
        )
        if not touches_border and area < max_hole_area:
            filled_mask[labels_inv == i] = 255

    return filled_mask


def smooth_edges(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Smooth edges and seal micro gaps via morphological closing."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def iter_image_files(folder: Path, exts: Iterable[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    exts = {e.lower() for e in exts}
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def refine_rgb_mask_image(img_bgr: np.ndarray) -> np.ndarray:
    """Refine an RGB mask image by binarizing and applying cleanup steps.

    Assumes foreground is light/white on dark background. Converts to grayscale,
    thresholds to binary, keeps largest component, fills small black holes,
    re-keeps largest, and smooths edges. Returns a single-channel uint8 mask.
    """
    if img_bgr is None:
        raise ValueError("Input image is None")

    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr

    # Binarize
    _, mask = cv2.threshold(gray.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)

    # Cleanup sequence
    largest = keep_largest_component(mask)
    filled = fill_black_islands(largest, max_hole_area=2000)
    largest2 = keep_largest_component(filled)
    cleaned = smooth_edges(largest2, kernel_size=5)
    return cleaned


def process_folder(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    ok = 0

    for img_path in iter_image_files(input_dir):
        total += 1
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[skip] Failed to read: {img_path}")
                continue
            cleaned = refine_rgb_mask_image(img)
            out_path = output_dir / img_path.name
            cv2.imwrite(str(out_path), cleaned)
            ok += 1
            print(f"[ok] {img_path} -> {out_path}")
        except Exception as e:
            print(f"[error] {img_path}: {e}")

    print(f"Done. Processed {ok}/{total} images. Output: {output_dir}")


if __name__ == "__main__":
    # Hardcoded folders: change as needed. Use raw strings on Windows if using absolute paths.
    # Example:
    # INPUT_DIR = Path(r"C:\Users\Staff\...\rgb_masks")
    # OUTPUT_DIR = Path(r"C:\Users\Staff\...\rgb_masks_refined")

    # Default relative paths
    INPUT_DIR = Path(r"C:\Users\Staff\OneDrive - University of Greenwich\HandsDatasets\handRGBD\eval\rgb_hand_mask")
    OUTPUT_DIR = Path(r"C:\Users\Staff\OneDrive - University of Greenwich\HandsDatasets\handRGBD\eval\rgb_hand_mask_refined")

    if not INPUT_DIR.exists():
        raise SystemExit(f"Input folder not found: {INPUT_DIR}")

    process_folder(INPUT_DIR, OUTPUT_DIR)

