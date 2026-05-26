import argparse
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from PIL import Image


DEFAULT_RGB_EXTENSIONS = (".jpg", ".jpeg")
DEFAULT_DEPTH_EXTENSION = ".tiff"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply binary hand masks to RGB JPGs and matching depth TIFFs, then "
            "export masked full-size images plus cropped hand regions."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder containing paired RGB JPGs and depth TIFFs.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Folder containing <stem>_mask.png files. Defaults to <input-dir>/masks.",
    )
    parser.add_argument(
        "--crop-padding",
        type=float,
        default=0.03,
        help="Extra padding around the mask bounding box as a fraction of width/height.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    return parser.parse_args()


def iter_rgb_images(folder: Path) -> Iterable[Path]:
    allowed = {ext.lower() for ext in DEFAULT_RGB_EXTENSIONS}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in allowed:
            yield path


def resolve_mask_path(mask_dir: Path, stem: str) -> Optional[Path]:
    candidates = (
        mask_dir / f"{stem}_mask.png",
        mask_dir / f"{stem}.png",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def ensure_mask_shape(mask_u8: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    image_h, image_w = image_shape[:2]
    if mask_u8.shape[:2] == (image_h, image_w):
        return mask_u8
    return cv2.resize(mask_u8, (image_w, image_h), interpolation=cv2.INTER_NEAREST)


def apply_rgb_mask(image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    binary_mask = np.where(mask_u8 > 0, 255, 0).astype(np.uint8)
    return cv2.bitwise_and(image_bgr, image_bgr, mask=binary_mask)


def apply_depth_mask(depth: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    binary_mask = mask_u8 > 0
    masked_depth = np.zeros_like(depth)
    masked_depth[binary_mask] = depth[binary_mask]
    return masked_depth.astype(np.float32, copy=False)


def find_mask_bbox(mask_u8: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def pad_bbox(
    bbox: tuple[int, int, int, int],
    image_w: int,
    image_h: int,
    padding_fraction: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad_x = int(round(box_w * padding_fraction))
    pad_y = int(round(box_h * padding_fraction))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(image_w, x2 + pad_x),
        min(image_h, y2 + pad_y),
    )


def scale_bbox(
    bbox: tuple[int, int, int, int],
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    sx = dst_w / max(1, src_w)
    sy = dst_h / max(1, src_h)
    return (
        max(0, min(dst_w, int(np.floor(x1 * sx)))),
        max(0, min(dst_h, int(np.floor(y1 * sy)))),
        max(0, min(dst_w, int(np.ceil(x2 * sx)))),
        max(0, min(dst_h, int(np.ceil(y2 * sy)))),
    )


def save_depth_tiff(path: Path, depth: np.ndarray) -> None:
    Image.fromarray(depth.astype(np.float32), mode="F").save(path)


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.expanduser()
    mask_dir = (args.mask_dir or (input_dir / "masks")).expanduser()

    rgb_masked_dir = input_dir / "rgb_masked"
    depth_masked_dir = input_dir / "depth_masked"
    rgb_crops_dir = input_dir / "rgb_crops"
    depth_crops_dir = input_dir / "depth_crops"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask folder not found: {mask_dir}")

    for folder in (rgb_masked_dir, depth_masked_dir, rgb_crops_dir, depth_crops_dir):
        folder.mkdir(parents=True, exist_ok=True)

    processed = 0
    saved = 0
    skipped_existing = 0
    missing_masks = 0
    missing_depth = 0
    failed_reads = 0
    empty_masks = 0

    for rgb_path in iter_rgb_images(input_dir):
        processed += 1
        stem = rgb_path.stem
        depth_path = input_dir / f"{stem}{DEFAULT_DEPTH_EXTENSION}"
        mask_path = resolve_mask_path(mask_dir, stem)

        rgb_masked_path = rgb_masked_dir / f"{stem}_masked.png"
        depth_masked_path = depth_masked_dir / f"{stem}_masked.tiff"
        rgb_crop_path = rgb_crops_dir / f"{stem}_crop.png"
        depth_crop_path = depth_crops_dir / f"{stem}_crop.tiff"

        if all(
            path.exists()
            for path in (rgb_masked_path, depth_masked_path, rgb_crop_path, depth_crop_path)
        ) and not args.overwrite:
            skipped_existing += 1
            print(f"[skip] {stem}: outputs already exist")
            continue

        if mask_path is None:
            missing_masks += 1
            print(f"[warn] {stem}: missing mask {stem}_mask.png or {stem}.png")
            continue

        if not depth_path.exists():
            missing_depth += 1
            print(f"[warn] {stem}: missing depth file {depth_path.name}")
            continue

        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        mask_u8 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if rgb_bgr is None or mask_u8 is None:
            failed_reads += 1
            print(f"[warn] {stem}: failed to read RGB or mask")
            continue

        try:
            depth = np.array(Image.open(depth_path), dtype=np.float32)
        except Exception as exc:
            failed_reads += 1
            print(f"[warn] {stem}: failed to read depth ({exc})")
            continue

        rgb_mask = ensure_mask_shape(mask_u8, rgb_bgr.shape)
        depth_mask = ensure_mask_shape(mask_u8, depth.shape)

        rgb_bbox = find_mask_bbox(rgb_mask)
        if rgb_bbox is None:
            empty_masks += 1
            print(f"[warn] {stem}: empty mask")
            continue

        rgb_bbox = pad_bbox(
            rgb_bbox,
            image_w=rgb_bgr.shape[1],
            image_h=rgb_bgr.shape[0],
            padding_fraction=args.crop_padding,
        )
        depth_bbox = scale_bbox(
            rgb_bbox,
            src_w=rgb_bgr.shape[1],
            src_h=rgb_bgr.shape[0],
            dst_w=depth.shape[1],
            dst_h=depth.shape[0],
        )

        rgb_masked = apply_rgb_mask(rgb_bgr, rgb_mask)
        depth_masked = apply_depth_mask(depth, depth_mask)

        x1, y1, x2, y2 = rgb_bbox
        dx1, dy1, dx2, dy2 = depth_bbox
        rgb_crop = rgb_masked[y1:y2, x1:x2]
        depth_crop = depth_masked[dy1:dy2, dx1:dx2]

        if rgb_crop.size == 0 or depth_crop.size == 0:
            empty_masks += 1
            print(f"[warn] {stem}: empty crop after bbox scaling")
            continue

        if not cv2.imwrite(str(rgb_masked_path), rgb_masked):
            failed_reads += 1
            print(f"[warn] {stem}: failed to write RGB masked image")
            continue

        save_depth_tiff(depth_masked_path, depth_masked)

        if not cv2.imwrite(str(rgb_crop_path), rgb_crop):
            failed_reads += 1
            print(f"[warn] {stem}: failed to write RGB crop")
            continue

        save_depth_tiff(depth_crop_path, depth_crop)

        saved += 1
        print(
            f"[ok] {stem}: "
            f"{rgb_masked_path.name}, {depth_masked_path.name}, "
            f"{rgb_crop_path.name}, {depth_crop_path.name}"
        )

    print(
        "Finished."
        f" Processed={processed}"
        f" Saved={saved}"
        f" SkippedExisting={skipped_existing}"
        f" MissingMasks={missing_masks}"
        f" MissingDepth={missing_depth}"
        f" FailedReads={failed_reads}"
        f" EmptyMasks={empty_masks}"
    )


if __name__ == "__main__":
    main()
