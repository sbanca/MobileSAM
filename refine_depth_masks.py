import cv2
import numpy as np
from pathlib import Path
from typing import Iterable


def refine_depth_mask(mask: np.ndarray) -> np.ndarray:
    """Fill holes in a binary/grayscale mask using flood fill.

    Expects a single-channel mask where foreground is non-zero. Returns a
    single-channel mask with interior holes filled.
    """
    if mask is None:
        raise ValueError("Input mask is None")

    # Ensure single-channel uint8
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Copy for flood fill and build working mask (2 px border per OpenCV API)
    im_floodfill = mask.copy()
    h, w = mask.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)

    # Flood fill from a corner assumed to be background
    cv2.floodFill(im_floodfill, flood_mask, (0, 0), 255)

    # Invert the flood-filled result and combine with original to fill holes
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    filled_mask = mask | im_floodfill_inv
    return filled_mask


def iter_image_files(folder: Path, exts: Iterable[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    exts = {e.lower() for e in exts}
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def process_folder(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    count_total = 0
    count_ok = 0

    for img_path in iter_image_files(input_dir):
        count_total += 1
        try:
            mask = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"[skip] Failed to read: {img_path}")
                continue
            filled_mask = refine_depth_mask(mask)

            out_path = output_dir / img_path.name
            cv2.imwrite(str(out_path), filled_mask)
            count_ok += 1
            print(f"[ok] {img_path} -> {out_path}")
        except Exception as e:
            print(f"[error] {img_path}: {e}")

    print(f"Done. Processed {count_ok}/{count_total} images. Output: {output_dir}")


if __name__ == "__main__":
    # Hardcoded folders: change as needed. Use raw strings on Windows.
    INPUT_DIR = Path(r"C:\Users\Staff\OneDrive - University of Greenwich\HandsDatasets\handRGBD\train\mask")  # folder containing input masks
    OUTPUT_DIR = Path(r"C:\Users\Staff\OneDrive - University of Greenwich\HandsDatasets\handRGBD\train\masks_refined")  # folder to save refined masks

    if not INPUT_DIR.exists():
        raise SystemExit(f"Input folder not found: {INPUT_DIR}")

    process_folder(INPUT_DIR, OUTPUT_DIR)
