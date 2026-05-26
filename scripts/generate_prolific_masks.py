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

from mobile_sam.automatic_mask_generator import SamAutomaticMaskGenerator
from mobile_sam.build_sam import sam_model_registry
from mobile_sam.predictor import SamPredictor


DEFAULT_DATASET_ROOT = Path(
    r"C:\Users\Staff\OneDrive - University of Greenwich\KE4ImpactFund\prolific_dataset"
)
DEFAULT_CSV_NAME = "prolific_image_landmarks.csv"
DEFAULT_CHECKPOINT = Path("weights") / "mobile_sam.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_MASK_SUBDIR = "masks"
DEFAULT_MASK_PREFIX = "Mask_"
DEFAULT_MASK_EXTENSION = ".png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-generate hand masks for the KE4ImpactFund Prolific dataset. "
            "Existing masks are skipped by default."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root containing participants/ and prolific_image_landmarks.csv.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Landmark CSV path. Defaults to <root>/prolific_image_landmarks.csv.",
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
        "--mask-subdir",
        type=str,
        default=DEFAULT_MASK_SUBDIR,
        help="Subfolder inside each participant folder for masks. Use empty string to save beside images.",
    )
    parser.add_argument(
        "--mask-prefix",
        type=str,
        default=DEFAULT_MASK_PREFIX,
        help="Prefix for output mask filenames.",
    )
    parser.add_argument(
        "--output-extension",
        type=str,
        default=DEFAULT_MASK_EXTENSION,
        help="Output mask extension.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mask files instead of skipping them.",
    )
    parser.add_argument(
        "--no-automatic-fallback",
        action="store_true",
        help="Do not use automatic mask generation when CSV landmarks are missing or fail.",
    )
    parser.add_argument(
        "--include-failed-detections",
        action="store_true",
        help="Use CSV landmark rows even when has_detection is not true.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of images to process after skip checks.",
    )
    parser.add_argument(
        "--save-overlays",
        action="store_true",
        help="Save red overlay previews beside generated masks.",
    )
    return parser.parse_args()


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


def parse_landmarks_px(raw_value: str) -> Optional[np.ndarray]:
    parsed = parse_json_like(raw_value)
    if parsed is None:
        return None

    points = np.asarray(parsed, dtype=np.float32)
    if points.size == 0 or points.ndim != 2 or points.shape[1] < 2:
        return None

    points = points[:, :2]
    points = points[np.isfinite(points).all(axis=1)]
    if points.size == 0:
        return None

    _, unique_indices = np.unique(points, axis=0, return_index=True)
    return points[np.sort(unique_indices)]


def row_has_detection(row: dict) -> bool:
    value = str(row.get("has_detection", "")).strip().lower()
    return value in {"1", "true", "yes", "y"}


def resolve_csv_path(dataset_root: Path, csv_path: Optional[Path]) -> Path:
    if csv_path is None:
        return dataset_root / DEFAULT_CSV_NAME
    if csv_path.is_absolute():
        return csv_path
    return dataset_root / csv_path


def resolve_image_path(row: dict, dataset_root: Path) -> Optional[Path]:
    candidates: list[Path] = []

    image_path_value = str(row.get("image_path", "")).strip()
    relative_path_value = str(row.get("relative_path", "")).strip()

    if image_path_value:
        path = Path(image_path_value)
        candidates.append(path if path.is_absolute() else dataset_root / path)

    if relative_path_value:
        candidates.append(dataset_root / "participants" / relative_path_value)
        candidates.append(dataset_root / relative_path_value)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def iter_csv_entries(
    csv_path: Path,
    dataset_root: Path,
    include_failed_detections: bool,
) -> Iterable[tuple[Path, Optional[np.ndarray], bool]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {csv_path}")

        required = {"image_path", "relative_path", "rgb_landmarks_px", "has_detection"}
        missing = required.difference(reader.fieldnames)
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"CSV file is missing required columns: {missing_str}")

        for row in reader:
            has_detection = row_has_detection(row)
            if not has_detection and not include_failed_detections:
                continue

            image_path = resolve_image_path(row, dataset_root)
            if image_path is None:
                print(
                    f"[WARN] Could not resolve image for "
                    f"relative_path='{row.get('relative_path', '')}'"
                )
                continue

            landmarks = parse_landmarks_px(str(row.get("rgb_landmarks_px", "")))
            yield image_path, landmarks, has_detection


def build_output_path(
    image_path: Path,
    mask_subdir: str,
    mask_prefix: str,
    output_extension: str,
) -> Path:
    output_name = f"{mask_prefix}{image_path.stem}{output_extension}"
    if mask_subdir:
        return image_path.parent / mask_subdir / output_name
    return image_path.with_name(output_name)


def load_model(checkpoint: Path, device: Optional[str]):
    import torch

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    build_fn = sam_model_registry["vit_t"]
    model = build_fn(checkpoint=str(checkpoint))
    model.to(device)
    model.eval()
    return model, device


def clamp_points(points_xy: np.ndarray, image_shape: Sequence[int]) -> np.ndarray:
    image_h, image_w = image_shape[:2]
    clipped = points_xy.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, max(image_w - 1, 0))
    clipped[:, 1] = np.clip(clipped[:, 1], 0, max(image_h - 1, 0))
    return clipped


def select_best_prompt_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    points_xy: np.ndarray,
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


def predict_from_landmarks(
    predictor: SamPredictor,
    image_bgr: np.ndarray,
    landmarks_px: np.ndarray,
) -> Optional[np.ndarray]:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    point_coords = clamp_points(landmarks_px, image_bgr.shape)
    point_labels = np.ones(point_coords.shape[0], dtype=np.int32)
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    best_mask = select_best_prompt_mask(masks, scores, point_coords)
    if best_mask is None:
        return None
    return best_mask.astype(bool)


def select_best_automatic_mask(masks: Sequence[dict], shape: Sequence[int]) -> Optional[np.ndarray]:
    if not masks:
        return None
    height, width = shape
    max_area = float(height * width)

    def score(mask: dict) -> float:
        segmentation = mask.get("segmentation")
        if segmentation is None:
            return -np.inf
        area = float(mask.get("area", segmentation.astype(bool).sum()))
        iou = float(mask.get("predicted_iou", 0.0))
        bbox = mask.get("bbox")
        center_bonus = 0.0
        if bbox is not None:
            x, y, w_box, h_box = bbox
            cx = x + w_box / 2.0
            cy = y + h_box / 2.0
            dx = abs(cx - width / 2.0) / max(1.0, width / 2.0)
            dy = abs(cy - height / 2.0) / max(1.0, height / 2.0)
            center_bonus = max(0.0, 1.0 - 0.5 * (dx + dy))
        area_term = area / max_area
        return iou * 0.7 + area_term * 0.2 + center_bonus * 0.1

    best = max(masks, key=score)
    segmentation = best.get("segmentation")
    if segmentation is None:
        return None
    return segmentation.astype(bool)


def predict_automatic(generator: SamAutomaticMaskGenerator, image_bgr: np.ndarray) -> Optional[np.ndarray]:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    masks = generator.generate(image_rgb)
    return select_best_automatic_mask(masks, image_bgr.shape[:2])


def save_mask(mask: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_u8 = np.where(mask, 255, 0).astype(np.uint8)
    if not cv2.imwrite(str(output_path), mask_u8):
        raise RuntimeError(f"Failed to write mask: {output_path}")


def build_overlay(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy().astype(np.float32)
    mask_bool = mask.astype(bool)
    tint = np.zeros_like(overlay)
    tint[..., 2] = 255
    overlay[mask_bool] = overlay[mask_bool] * 0.55 + tint[mask_bool] * 0.45
    return overlay.clip(0, 255).astype(np.uint8)


def main() -> None:
    args = parse_args()

    dataset_root = args.root.expanduser()
    csv_path = resolve_csv_path(dataset_root, args.csv)
    output_extension = normalize_extension(args.output_extension)
    mask_subdir = args.mask_subdir.strip()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    model, device = load_model(args.checkpoint, args.device)
    predictor = SamPredictor(model)
    automatic_generator = None
    if not args.no_automatic_fallback:
        automatic_generator = SamAutomaticMaskGenerator(model)

    print(f"Using device: {device}")
    print(f"Reading CSV: {csv_path}")

    visited: set[Path] = set()
    seen = 0
    processed = 0
    saved = 0
    skipped_existing = 0
    skipped_duplicate = 0
    missing_landmarks = 0
    automatic_fallbacks = 0
    failed = 0

    entries = iter_csv_entries(
        csv_path=csv_path,
        dataset_root=dataset_root,
        include_failed_detections=args.include_failed_detections,
    )

    for image_path, landmarks_px, _ in entries:
        resolved = image_path.resolve()
        if resolved in visited:
            skipped_duplicate += 1
            continue
        visited.add(resolved)
        seen += 1

        output_path = build_output_path(
            image_path,
            mask_subdir,
            args.mask_prefix,
            output_extension,
        )
        if output_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        if args.limit is not None and processed >= args.limit:
            break
        processed += 1

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            failed += 1
            print(f"[WARN] Failed to read image: {image_path}")
            continue

        mask = None
        method = "landmarks"
        if landmarks_px is not None and len(landmarks_px) > 0:
            try:
                mask = predict_from_landmarks(predictor, image_bgr, landmarks_px)
            except Exception as exc:
                print(f"[WARN] Landmark prediction failed for {image_path.name}: {exc}")
        else:
            missing_landmarks += 1

        if (mask is None or not mask.any()) and automatic_generator is not None:
            method = "automatic"
            automatic_fallbacks += 1
            try:
                mask = predict_automatic(automatic_generator, image_bgr)
            except Exception as exc:
                print(f"[WARN] Automatic prediction failed for {image_path.name}: {exc}")

        if mask is None or not mask.any():
            failed += 1
            print(f"[WARN] Empty mask for {image_path.name}")
            continue

        try:
            save_mask(mask, output_path)
        except Exception as exc:
            failed += 1
            print(f"[WARN] {exc}")
            continue

        saved += 1
        print(f"[OK] {image_path.parent.name}/{image_path.name} -> {output_path.name} ({method})")

        if args.save_overlays:
            overlay_path = output_path.with_name(f"{output_path.stem}_overlay{output_path.suffix}")
            overlay = build_overlay(image_bgr, mask)
            if not cv2.imwrite(str(overlay_path), overlay):
                print(f"[WARN] Failed to write overlay: {overlay_path}")

    print(
        "Finished."
        f" Seen={seen}"
        f" Processed={processed}"
        f" Saved={saved}"
        f" SkippedExisting={skipped_existing}"
        f" SkippedDuplicateRows={skipped_duplicate}"
        f" MissingLandmarks={missing_landmarks}"
        f" AutomaticFallbacks={automatic_fallbacks}"
        f" Failed={failed}"
    )


if __name__ == "__main__":
    main()
