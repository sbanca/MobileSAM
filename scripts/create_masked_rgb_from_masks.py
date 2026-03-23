import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


DEFAULT_DATASET_ROOT = Path(
    r"C:\Users\Staff\OneDrive - University of Greenwich\HandsDatasets\HaGRIDv2_stop_inverted"
)
DEFAULT_RGB_DIR = "rgb"
DEFAULT_MASK_DIR = "masks"
DEFAULT_OUTPUT_DIR = "rgb_masked"
DEFAULT_MASK_SUFFIX = "Mask"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create masked RGB images by applying binary masks from a mask folder "
            "to matching RGB images."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root containing the RGB, mask, and output folders.",
    )
    parser.add_argument(
        "--rgb-dir",
        type=str,
        default=DEFAULT_RGB_DIR,
        help="RGB image directory under dataset root.",
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        default=DEFAULT_MASK_DIR,
        help="Mask directory under dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory under dataset root for masked RGB images.",
    )
    parser.add_argument(
        "--mask-suffix",
        type=str,
        default=DEFAULT_MASK_SUFFIX,
        help="Suffix appended to the RGB stem when resolving mask filenames.",
    )
    parser.add_argument(
        "--output-extension",
        type=str,
        default=".png",
        help="Output file extension for masked RGB images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing masked RGB images.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of RGB images to inspect.",
    )
    parser.add_argument(
        "--skip-missing-masks",
        action="store_true",
        help="Skip RGB images without a matching mask instead of warning for each one.",
    )
    return parser.parse_args()


def normalize_extension(value: str) -> str:
    value = value.strip()
    if not value:
        return ".png"
    if not value.startswith("."):
        value = f".{value}"
    return value.lower()


def iter_rgb_images(rgb_dir: Path) -> Iterable[Path]:
    for path in sorted(rgb_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def resolve_mask_path(image_path: Path, mask_dir: Path, mask_suffix: str) -> Path:
    return mask_dir / f"{image_path.stem}{mask_suffix}.png"


def build_output_path(image_path: Path, output_dir: Path, output_extension: str) -> Path:
    return output_dir / f"{image_path.stem}{output_extension}"


def ensure_mask_shape(mask_u8: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    image_h, image_w = image_shape[:2]
    if mask_u8.shape[:2] == (image_h, image_w):
        return mask_u8
    return cv2.resize(mask_u8, (image_w, image_h), interpolation=cv2.INTER_NEAREST)


def apply_mask(image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    binary_mask = np.where(mask_u8 > 0, 255, 0).astype(np.uint8)
    return cv2.bitwise_and(image_bgr, image_bgr, mask=binary_mask)


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root.expanduser()
    rgb_dir = dataset_root / args.rgb_dir
    mask_dir = dataset_root / args.mask_dir
    output_dir = dataset_root / args.output_dir
    output_extension = normalize_extension(args.output_extension)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    processed = 0
    saved = 0
    skipped_existing = 0
    missing_masks = 0
    failed_reads = 0
    failed_writes = 0
    resized_masks = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in iter_rgb_images(rgb_dir):
        if args.limit is not None and processed >= args.limit:
            break
        processed += 1

        mask_path = resolve_mask_path(image_path, mask_dir, args.mask_suffix)
        if not mask_path.exists():
            missing_masks += 1
            if not args.skip_missing_masks:
                print(f"[WARN] Missing mask for {image_path.name}: {mask_path.name}")
            continue

        output_path = build_output_path(image_path, output_dir, output_extension)
        if output_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask_u8 = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image_bgr is None or mask_u8 is None:
            failed_reads += 1
            print(f"[WARN] Failed to read image or mask: {image_path.name}")
            continue

        adjusted_mask = ensure_mask_shape(mask_u8, image_bgr.shape)
        if adjusted_mask.shape != mask_u8.shape:
            resized_masks += 1

        masked_bgr = apply_mask(image_bgr, adjusted_mask)

        if not cv2.imwrite(str(output_path), masked_bgr):
            failed_writes += 1
            print(f"[WARN] Failed to write masked RGB: {output_path}")
            continue

        saved += 1
        print(f"[OK] {image_path.name} -> {output_path.name}")

    print(
        "Finished."
        f" Processed={processed}"
        f" Saved={saved}"
        f" SkippedExisting={skipped_existing}"
        f" MissingMasks={missing_masks}"
        f" FailedReads={failed_reads}"
        f" FailedWrites={failed_writes}"
        f" ResizedMasks={resized_masks}"
    )


if __name__ == "__main__":
    main()
