import argparse
from pathlib import Path
import sys
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from handLandmarks.handLandmarksDetection import MediaPipeTaskHandLandmarkDetector
from mobile_sam.automatic_mask_generator import SamAutomaticMaskGenerator
from mobile_sam.build_sam import sam_model_registry
from mobile_sam.predictor import SamPredictor


DEFAULT_CHECKPOINT = Path("weights") / "mobile_sam.pt"
DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate binary hand masks for all JPG images in a folder using "
            "MediaPipe hand landmarks as MobileSAM prompts."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder containing input images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Folder for output masks. Defaults to <input-dir>/masks.",
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
        "--box-margin",
        type=int,
        default=24,
        help="Extra pixels added around the landmark-derived prompt box.",
    )
    parser.add_argument(
        "--save-overlays",
        action="store_true",
        help="Save red overlay previews under <output-dir>/overlays.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mask files.",
    )
    parser.add_argument(
        "--fallback-auto",
        action="store_true",
        help="Fall back to automatic mask generation when landmark detection fails.",
    )
    return parser.parse_args()


def iter_images(folder: Path, extensions: Sequence[str]) -> Iterable[Path]:
    allowed = {ext.lower() for ext in extensions}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in allowed:
            yield path


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


def select_best_prompted_mask(
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


def select_best_automatic_mask(
    masks: list[dict], image_shape: Sequence[int]
) -> Optional[np.ndarray]:
    if not masks:
        return None

    image_h, image_w = image_shape[:2]
    max_area = float(image_h * image_w)
    best_mask = None
    best_score = float("-inf")

    for mask_info in masks:
        segmentation = mask_info.get("segmentation")
        if segmentation is None:
            continue
        bbox = mask_info.get("bbox")
        center_bonus = 0.0
        if bbox is not None:
            x, y, w_box, h_box = bbox
            cx = x + w_box / 2.0
            cy = y + h_box / 2.0
            dx = abs(cx - image_w / 2.0) / max(1.0, image_w / 2.0)
            dy = abs(cy - image_h / 2.0) / max(1.0, image_h / 2.0)
            center_bonus = max(0.0, 1.0 - 0.5 * (dx + dy))
        area = float(mask_info.get("area", float(np.asarray(segmentation).sum())))
        predicted_iou = float(mask_info.get("predicted_iou", 0.0))
        score = predicted_iou * 0.7 + (area / max_area) * 0.2 + center_bonus * 0.1
        if score > best_score:
            best_score = score
            best_mask = np.asarray(segmentation, dtype=bool)

    return best_mask


def keep_largest_component(mask_u8: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


def fill_small_holes(mask_u8: np.ndarray, max_hole_area: int = 2000) -> np.ndarray:
    inverse = cv2.bitwise_not(mask_u8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    filled = mask_u8.copy()

    for idx in range(1, num_labels):
        x, y, w_box, h_box, area = stats[idx]
        touches_border = (
            x <= 0
            or y <= 0
            or (x + w_box) >= inverse.shape[1]
            or (y + h_box) >= inverse.shape[0]
        )
        if not touches_border and area <= max_hole_area:
            filled[labels == idx] = 255

    return filled


def cleanup_mask(mask_bool: np.ndarray) -> np.ndarray:
    mask_u8 = (mask_bool.astype(np.uint8)) * 255
    mask_u8 = keep_largest_component(mask_u8)
    mask_u8 = fill_small_holes(mask_u8, max_hole_area=2000)
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)


def build_overlay(image_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy().astype(np.float32)
    mask_bool = mask_u8 > 0
    tint = np.zeros_like(overlay)
    tint[..., 2] = 255
    overlay[mask_bool] = overlay[mask_bool] * 0.55 + tint[mask_bool] * 0.45
    return overlay.clip(0, 255).astype(np.uint8)


def load_model(device: Optional[str], checkpoint: Path):
    import torch

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = sam_model_registry["vit_t"](checkpoint=str(checkpoint))
    model.to(device)
    model.eval()
    return model, device


def generate_mask_from_landmarks(
    predictor: SamPredictor,
    detector: MediaPipeTaskHandLandmarkDetector,
    image_bgr: np.ndarray,
    box_margin: int,
) -> Optional[np.ndarray]:
    _, landmarks_px = detector.detect(image_bgr)
    if getattr(landmarks_px, "size", 0) == 0:
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    point_coords = clamp_points(landmarks_px[:, :2].astype(np.float32), image_bgr.shape)
    point_labels = np.ones(point_coords.shape[0], dtype=np.int32)
    box = landmarks_to_box(point_coords, image_bgr.shape, box_margin)
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=True,
    )
    best_mask = select_best_prompted_mask(masks, scores, point_coords)
    if best_mask is None:
        return None
    return cleanup_mask(best_mask.astype(bool))


def generate_mask_automatic(
    generator: SamAutomaticMaskGenerator,
    image_bgr: np.ndarray,
) -> Optional[np.ndarray]:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    masks = generator.generate(image_rgb)
    best_mask = select_best_automatic_mask(masks, image_rgb.shape)
    if best_mask is None:
        return None
    return cleanup_mask(best_mask)


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.expanduser()
    output_dir = (args.output_dir or (input_dir / "masks")).expanduser()
    overlay_dir = output_dir / "overlays"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    image_paths = list(iter_images(input_dir, DEFAULT_IMAGE_EXTENSIONS))
    if not image_paths:
        raise FileNotFoundError(f"No JPG images found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    model, device = load_model(args.device, args.checkpoint)
    predictor = SamPredictor(model)
    detector = MediaPipeTaskHandLandmarkDetector()
    automatic_generator = SamAutomaticMaskGenerator(model) if args.fallback_auto else None

    processed = 0
    saved = 0
    skipped_existing = 0
    landmark_failures = 0
    automatic_fallbacks = 0
    failed = 0

    print(f"Using device: {device}")
    for image_path in image_paths:
        processed += 1
        output_path = output_dir / f"{image_path.stem}_mask.png"
        overlay_path = overlay_dir / f"{image_path.stem}_overlay.png"

        if output_path.exists() and not args.overwrite:
            skipped_existing += 1
            print(f"[skip] {image_path.name} -> {output_path.name} already exists")
            continue

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            failed += 1
            print(f"[fail] {image_path.name}: failed to read image")
            continue

        try:
            mask_u8 = generate_mask_from_landmarks(
                predictor,
                detector,
                image_bgr,
                args.box_margin,
            )
        except Exception as exc:
            landmark_failures += 1
            mask_u8 = None
            print(f"[warn] {image_path.name}: landmark prompt failed ({exc})")

        if mask_u8 is None and automatic_generator is not None:
            try:
                mask_u8 = generate_mask_automatic(automatic_generator, image_bgr)
                if mask_u8 is not None:
                    automatic_fallbacks += 1
            except Exception as exc:
                print(f"[warn] {image_path.name}: automatic fallback failed ({exc})")

        if mask_u8 is None or not mask_u8.any():
            failed += 1
            print(f"[fail] {image_path.name}: no mask generated")
            continue

        if not cv2.imwrite(str(output_path), mask_u8):
            failed += 1
            print(f"[fail] {image_path.name}: failed to write {output_path}")
            continue

        if args.save_overlays:
            overlay = build_overlay(image_bgr, mask_u8)
            if not cv2.imwrite(str(overlay_path), overlay):
                print(f"[warn] {image_path.name}: failed to write overlay")

        saved += 1
        print(f"[ok] {image_path.name} -> {output_path.name}")

    print(
        "Done."
        f" Processed={processed}"
        f" Saved={saved}"
        f" SkippedExisting={skipped_existing}"
        f" LandmarkFailures={landmark_failures}"
        f" AutomaticFallbacks={automatic_fallbacks}"
        f" Failed={failed}"
        f" OutputDir={output_dir}"
    )


if __name__ == "__main__":
    main()
