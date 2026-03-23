import argparse
import ast
import csv
import json
from pathlib import Path
import sys
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mobile_sam.build_sam import sam_model_registry
from mobile_sam.predictor import SamPredictor


DEFAULT_DATASET_ROOT = Path(
    r"C:\Users\Staff\OneDrive - University of Greenwich\HandsDatasets\HaGRIDv2_stop_inverted"
)
DEFAULT_CSV_NAME = "reference_hagrid_stop_inverted.csv"
DEFAULT_IMAGE_DIR = "rgb"
DEFAULT_CHECKPOINT = Path("weights") / "mobile_sam.pt"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate binary hand masks for the HaGRIDv2 stop_inverted export "
            "using hand_landmarks from reference_hagrid_stop_inverted.csv as "
            "MobileSAM click prompts."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root containing reference_hagrid_stop_inverted.csv and rgb/.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV path. Defaults to <dataset-root>/reference_hagrid_stop_inverted.csv.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=DEFAULT_IMAGE_DIR,
        help="Image directory under the dataset root.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="MobileSAM checkpoint path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device, for example cuda or cpu. Defaults to cuda if available.",
    )
    parser.add_argument(
        "--landmark-space",
        choices=("auto", "source_norm", "crop_norm", "pixel"),
        default="auto",
        help=(
            "Interpretation of hand_landmarks. "
            "'source_norm' expects values normalized to the original HaGRID frame, "
            "'crop_norm' expects values normalized to the exported crop, "
            "'pixel' expects pixel coordinates on the exported crop, and "
            "'auto' prefers source_norm when the needed metadata is present."
        ),
    )
    parser.add_argument(
        "--box-margin",
        type=int,
        default=25,
        help="Extra pixels added around the landmark-derived prompt box.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="Mask",
        help="Suffix inserted before the output extension.",
    )
    parser.add_argument(
        "--output-extension",
        type=str,
        default=".png",
        help="Output file extension for saved masks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for masks. Defaults to saving beside each image.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mask files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of CSV rows to process.",
    )
    parser.add_argument(
        "--save-overlays",
        action="store_true",
        help="Save a red overlay preview beside each generated mask.",
    )
    return parser.parse_args()


def resolve_csv_path(dataset_root: Path, csv_path: Optional[Path]) -> Path:
    if csv_path is not None:
        if csv_path.is_absolute():
            return csv_path
        return dataset_root / csv_path
    return dataset_root / DEFAULT_CSV_NAME


def normalize_extension(value: str) -> str:
    value = value.strip()
    if not value:
        return ".png"
    if not value.startswith("."):
        value = f".{value}"
    return value.lower()


def parse_json_like(raw_value: str) -> Optional[object]:
    raw_value = (raw_value or "").strip()
    if not raw_value or raw_value in {"[]", "nan", "None", "null"}:
        return None

    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(raw_value)
        except Exception:
            continue
    return None


def parse_landmarks(raw_value: str) -> Optional[np.ndarray]:
    parsed = parse_json_like(raw_value)
    if parsed is None:
        return None

    points = np.asarray(parsed, dtype=np.float32)
    if points.size == 0:
        return None
    if points.ndim != 2 or points.shape[1] < 2:
        return None

    points = points[:, :2]
    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]
    if points.size == 0:
        return None

    _, unique_indices = np.unique(points, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    return points[unique_indices]


def parse_xyxy(raw_value: str) -> Optional[np.ndarray]:
    parsed = parse_json_like(raw_value)
    if parsed is None:
        return None

    values = np.asarray(parsed, dtype=np.float32).reshape(-1)
    if values.size < 4:
        return None
    if not np.isfinite(values[:4]).all():
        return None
    return values[:4]


def points_are_normalized(points_xy: np.ndarray) -> bool:
    return bool(
        np.isfinite(points_xy).all()
        and np.min(points_xy) >= -1e-6
        and np.max(points_xy) <= 1.0 + 1e-6
    )


def clamp_points(points_xy: np.ndarray, image_shape: Sequence[int]) -> np.ndarray:
    image_h, image_w = image_shape[:2]
    clipped = points_xy.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, max(image_w - 1, 0))
    clipped[:, 1] = np.clip(clipped[:, 1], 0, max(image_h - 1, 0))
    return clipped


def landmarks_to_box(
    points_xy: np.ndarray, image_shape: Sequence[int], margin: int
) -> np.ndarray:
    image_h, image_w = image_shape[:2]
    x1 = max(0, int(np.floor(points_xy[:, 0].min())) - margin)
    y1 = max(0, int(np.floor(points_xy[:, 1].min())) - margin)
    x2 = min(image_w, int(np.ceil(points_xy[:, 0].max())) + margin + 1)
    y2 = min(image_h, int(np.ceil(points_xy[:, 1].max())) + margin + 1)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def crop_norm_landmarks_to_pixels(
    points_xy: np.ndarray, image_shape: Sequence[int]
) -> np.ndarray:
    image_h, image_w = image_shape[:2]
    points_px = points_xy.copy()
    points_px[:, 0] *= image_w
    points_px[:, 1] *= image_h
    return points_px


def source_norm_landmarks_to_crop_pixels(
    points_xy: np.ndarray, row: dict, image_shape: Sequence[int]
) -> np.ndarray:
    crop_xyxy = parse_xyxy(str(row.get("square_crop_xyxy_px", "")))
    if crop_xyxy is None:
        raise ValueError("Missing or invalid square_crop_xyxy_px")

    try:
        source_img_w = float(str(row.get("source_img_w", "")).strip())
        source_img_h = float(str(row.get("source_img_h", "")).strip())
    except ValueError as exc:
        raise ValueError("Missing or invalid source image dimensions") from exc

    if source_img_w <= 0 or source_img_h <= 0:
        raise ValueError("Source image dimensions must be positive")

    x1, y1, x2, y2 = crop_xyxy.tolist()
    crop_w = max(x2 - x1, 1.0)
    crop_h = max(y2 - y1, 1.0)
    image_h, image_w = image_shape[:2]

    points_px = points_xy.copy()
    points_px[:, 0] *= source_img_w
    points_px[:, 1] *= source_img_h

    points_px[:, 0] = (points_px[:, 0] - x1) / crop_w * image_w
    points_px[:, 1] = (points_px[:, 1] - y1) / crop_h * image_h
    return points_px


def landmarks_to_pixels(
    points_xy: np.ndarray,
    row: dict,
    image_shape: Sequence[int],
    landmark_space: str,
) -> np.ndarray:
    if landmark_space == "pixel":
        return points_xy.copy()

    if landmark_space == "crop_norm":
        if not points_are_normalized(points_xy):
            raise ValueError("Expected normalized crop landmarks in [0, 1]")
        return crop_norm_landmarks_to_pixels(points_xy, image_shape)

    if landmark_space == "source_norm":
        if not points_are_normalized(points_xy):
            raise ValueError("Expected source-normalized landmarks in [0, 1]")
        return source_norm_landmarks_to_crop_pixels(points_xy, row, image_shape)

    if points_are_normalized(points_xy):
        try:
            return source_norm_landmarks_to_crop_pixels(points_xy, row, image_shape)
        except ValueError:
            return crop_norm_landmarks_to_pixels(points_xy, image_shape)

    return points_xy.copy()


def resolve_image_path(row: dict, dataset_root: Path, image_dir: str) -> Optional[Path]:
    name = str(row.get("name", "")).strip()
    if not name:
        return None

    image_root = dataset_root / image_dir
    candidates: list[Path] = []

    name_path = Path(name)
    if name_path.suffix:
        candidates.append(image_root / name_path.name)
    else:
        for extension in IMAGE_EXTENSIONS:
            candidates.append(image_root / f"{name}{extension}")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_output_path(
    image_path: Path,
    suffix: str,
    output_extension: str,
    output_dir: Optional[Path],
    dataset_root: Path,
) -> Path:
    output_name = f"{image_path.stem}{suffix}{output_extension}"
    if output_dir is None:
        return image_path.with_name(output_name)

    if output_dir.is_absolute():
        return output_dir / output_name

    return (dataset_root / output_dir) / output_name


def build_overlay(image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy().astype(np.float32)
    mask_bool = mask_u8 > 0
    tint = np.zeros_like(overlay)
    tint[..., 2] = 255
    overlay[mask_bool] = overlay[mask_bool] * 0.55 + tint[mask_bool] * 0.45
    return overlay.clip(0, 255).astype(np.uint8)


def load_predictor(checkpoint: Path, device: Optional[str]) -> SamPredictor:
    import torch

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    build_fn = sam_model_registry["vit_t"]
    model = build_fn(checkpoint=str(checkpoint))
    model.to(device)
    model.eval()
    return SamPredictor(model)


def select_best_mask(
    masks: np.ndarray, scores: np.ndarray, points_xy: np.ndarray
) -> Optional[np.ndarray]:
    if masks.ndim != 3 or masks.shape[0] == 0:
        return None

    rounded = np.rint(points_xy).astype(np.int32)
    best_index = -1
    best_key: tuple[float, float, float] | None = None

    for idx, mask in enumerate(masks):
        inside = mask[rounded[:, 1], rounded[:, 0]]
        coverage = float(np.mean(inside.astype(np.float32)))
        area = float(mask.sum())
        key = (coverage, float(scores[idx]), -area)
        if best_key is None or key > best_key:
            best_key = key
            best_index = idx

    if best_index < 0:
        return None
    return masks[best_index]


def predict_mask(
    predictor: SamPredictor,
    image_bgr: np.ndarray,
    points_xy: np.ndarray,
    box_margin: int,
) -> Optional[np.ndarray]:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    point_coords = clamp_points(points_xy, image_bgr.shape)
    point_labels = np.ones(point_coords.shape[0], dtype=np.int32)
    box = landmarks_to_box(point_coords, image_bgr.shape, box_margin)

    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=True,
    )

    best_mask = select_best_mask(masks, scores, point_coords)
    if best_mask is None:
        return None
    return (best_mask.astype(np.uint8)) * 255


def iter_rows(csv_path: Path) -> Iterable[dict]:
    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {csv_path}")
        required = {
            "name",
            "hand_landmarks",
            "square_crop_xyxy_px",
            "source_img_w",
            "source_img_h",
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"CSV file is missing required columns: {missing_str}")
        yield from reader


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root.expanduser()
    csv_path = resolve_csv_path(dataset_root, args.csv)
    output_extension = normalize_extension(args.output_extension)
    image_dir_path = dataset_root / args.image_dir

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not image_dir_path.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir_path}")

    predictor = load_predictor(args.checkpoint, args.device)

    processed = 0
    saved = 0
    skipped_existing = 0
    skipped_missing_landmarks = 0
    skipped_bad_landmarks = 0
    missing_images = 0
    failed_predictions = 0

    for row in iter_rows(csv_path):
        if args.limit is not None and processed >= args.limit:
            break
        processed += 1

        points_xy = parse_landmarks(str(row.get("hand_landmarks", "")))
        if points_xy is None or len(points_xy) == 0:
            skipped_missing_landmarks += 1
            continue

        image_path = resolve_image_path(row, dataset_root, args.image_dir)
        if image_path is None:
            missing_images += 1
            print(f"[WARN] Could not resolve image for row name='{row.get('name', '')}'")
            continue

        output_path = build_output_path(
            image_path=image_path,
            suffix=args.output_suffix,
            output_extension=output_extension,
            output_dir=args.output_dir,
            dataset_root=dataset_root,
        )
        if output_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            missing_images += 1
            print(f"[WARN] Failed to read image: {image_path}")
            continue

        try:
            points_px = landmarks_to_pixels(
                points_xy=points_xy,
                row=row,
                image_shape=image_bgr.shape,
                landmark_space=args.landmark_space,
            )
        except ValueError as exc:
            skipped_bad_landmarks += 1
            print(f"[WARN] Invalid landmarks for {image_path.name}: {exc}")
            continue

        try:
            mask_u8 = predict_mask(
                predictor=predictor,
                image_bgr=image_bgr,
                points_xy=points_px,
                box_margin=args.box_margin,
            )
        except Exception as exc:
            failed_predictions += 1
            print(f"[WARN] MobileSAM failed for {image_path.name}: {exc}")
            continue

        if mask_u8 is None or not mask_u8.any():
            failed_predictions += 1
            print(f"[WARN] Empty mask for {image_path.name}")
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), mask_u8):
            failed_predictions += 1
            print(f"[WARN] Failed to write mask: {output_path}")
            continue
        saved += 1

        if args.save_overlays:
            overlay_path = output_path.with_name(
                f"{output_path.stem}_overlay{output_path.suffix}"
            )
            overlay = build_overlay(image_bgr, mask_u8)
            if not cv2.imwrite(str(overlay_path), overlay):
                print(f"[WARN] Failed to write overlay: {overlay_path}")

        print(f"[OK] {image_path.name} -> {output_path.name}")

    print(
        "Finished."
        f" Processed={processed}"
        f" Saved={saved}"
        f" SkippedExisting={skipped_existing}"
        f" SkippedMissingLandmarks={skipped_missing_landmarks}"
        f" SkippedBadLandmarks={skipped_bad_landmarks}"
        f" MissingImages={missing_images}"
        f" FailedPredictions={failed_predictions}"
    )


if __name__ == "__main__":
    main()
