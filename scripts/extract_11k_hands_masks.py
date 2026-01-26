import argparse
import csv
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


_SAM_PREDICTOR = None


DEFAULT_DATASET_ROOT = Path(
    r"C:\Users\Staff\OneDrive - University of Greenwich\HandsDatasets\11kHands"
)
DEFAULT_IMAGES_DIR_NAME = "Hands"
DEFAULT_MASKS_DIR_NAME = "Masks"
DEFAULT_CSV_NAME = "HandInfo.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate binary hand masks for the 11k Hands dataset by thresholding white "
            "background pixels."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root folder that contains HandInfo.csv and Hands/.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to HandInfo.csv (defaults to <dataset-root>/HandInfo.csv).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Path to images folder (defaults to <dataset-root>/Hands).",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        default=None,
        help="Output folder for masks (defaults to <dataset-root>/Masks).",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=".jpg,.jpeg,.png,.bmp",
        help="Comma-separated list of extensions to try when imageName has no suffix.",
    )
    parser.add_argument(
        "--white-min",
        type=int,
        default=220,
        help=(
            "Minimum channel value to treat a pixel as white background (0-255). "
            "Used with --bg-method simple."
        ),
    )
    parser.add_argument(
        "--white-tolerance",
        type=int,
        default=25,
        help=(
            "Max channel spread to still count as white (max - min). "
            "Used with --bg-method simple."
        ),
    )
    parser.add_argument(
        "--bg-method",
        choices=("simple", "border"),
        default="border",
        help=(
            "Background detection method. 'simple' uses a fixed white threshold. "
            "'border' adapts to lighting using border statistics (default)."
        ),
    )
    parser.add_argument(
        "--border-percentile",
        type=float,
        default=95.0,
        help="Percentile of border distance-to-white to classify background in border mode.",
    )
    parser.add_argument(
        "--border-margin",
        type=int,
        default=10,
        help="Border width (in pixels) to sample background statistics in border mode.",
    )
    parser.add_argument(
        "--refine-sam",
        action="store_true",
        help="Refine the initial mask using MobileSAM with the mask as input.",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=Path("weights") / "mobile_sam.pt",
        help="Path to MobileSAM checkpoint (default: weights/mobile_sam.pt).",
    )
    parser.add_argument(
        "--sam-device",
        type=str,
        default=None,
        help="Device for SAM (e.g. cuda, cpu). Defaults to cuda if available.",
    )
    parser.add_argument(
        "--sam-box-margin",
        type=int,
        default=10,
        help="Extra margin (pixels) added around the mask bbox before SAM refinement.",
    )
    parser.add_argument(
        "--sam-multimask",
        action="store_true",
        help="Request multiple SAM masks and select the one with highest IoU score.",
    )
    parser.add_argument(
        "--bbox-format",
        choices=("auto", "xywh", "xyxy"),
        default="auto",
        help=(
            "Format of bbox column values. 'xywh' uses (x,y,width,height). "
            "'xyxy' uses (x1,y1,x2,y2). 'auto' tries to infer."
        ),
    )
    parser.add_argument(
        "--use-bbox",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the CSV bbox to limit mask creation (default: False).",
    )
    parser.add_argument(
        "--close-kernel",
        type=int,
        default=0,
        help="Optional morphological closing kernel size (0 disables).",
    )
    parser.add_argument(
        "--open-kernel",
        type=int,
        default=0,
        help="Optional morphological opening kernel size (0 disables).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of rows to process.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help=(
            "Show each image with its generated mask (close the window to advance, "
            "press q to quit)."
        ),
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=10,
        help="Maximum number of previews to show when --preview is enabled.",
    )
    return parser.parse_args()


def _parse_extensions(extensions: str) -> Sequence[str]:
    exts = [ext.strip() for ext in extensions.split(",") if ext.strip()]
    normalized = []
    for ext in exts:
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.append(ext.lower())
    return normalized


def _resolve_image_path(
    image_name: str, images_dir: Path, extensions: Sequence[str]
) -> Optional[Path]:
    if not image_name:
        return None

    name_path = Path(image_name)
    if name_path.is_absolute():
        if name_path.exists():
            return name_path
        return None

    # If the relative path already exists as-is (relative to CWD), use it.
    if name_path.exists():
        return name_path

    candidate_roots = [images_dir]
    if images_dir.parent != images_dir:
        candidate_roots.append(images_dir.parent)

    candidates: Iterable[Path]
    if name_path.suffix:
        candidates = [root / name_path for root in candidate_roots]
    else:
        candidates = [
            root / f"{image_name}{ext}"
            for root in candidate_roots
            for ext in extensions
        ]

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _parse_bbox_values(bbox_str: str) -> Optional[Tuple[int, int, int, int]]:
    if not bbox_str:
        return None
    parts = [p.strip() for p in bbox_str.split(",")]
    if len(parts) != 4:
        return None
    try:
        values = tuple(int(float(p)) for p in parts)
    except ValueError:
        return None
    return values  # type: ignore[return-value]


def _infer_bbox(
    values: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    fmt: str,
) -> Tuple[int, int, int, int]:
    x1, y1, a, b = values

    def clamp_bbox(xmin: int, ymin: int, xmax: int, ymax: int) -> Tuple[int, int, int, int]:
        xmin = max(0, min(xmin, img_w - 1))
        ymin = max(0, min(ymin, img_h - 1))
        xmax = max(0, min(xmax, img_w))
        ymax = max(0, min(ymax, img_h))
        return xmin, ymin, xmax, ymax

    def bbox_from_xywh() -> Tuple[int, int, int, int]:
        return clamp_bbox(x1, y1, x1 + a, y1 + b)

    def bbox_from_xyxy() -> Tuple[int, int, int, int]:
        return clamp_bbox(x1, y1, a, b)

    if fmt == "xywh":
        return bbox_from_xywh()
    if fmt == "xyxy":
        return bbox_from_xyxy()

    bbox_xywh = bbox_from_xywh()
    bbox_xyxy = bbox_from_xyxy()

    def is_valid(bbox: Tuple[int, int, int, int]) -> bool:
        xmin, ymin, xmax, ymax = bbox
        return xmax > xmin and ymax > ymin

    valid_xywh = is_valid(bbox_xywh)
    valid_xyxy = is_valid(bbox_xyxy)

    if valid_xywh and not valid_xyxy:
        return bbox_xywh
    if valid_xyxy and not valid_xywh:
        return bbox_xyxy
    if valid_xywh and valid_xyxy:
        area_xywh = (bbox_xywh[2] - bbox_xywh[0]) * (bbox_xywh[3] - bbox_xywh[1])
        area_xyxy = (bbox_xyxy[2] - bbox_xyxy[0]) * (bbox_xyxy[3] - bbox_xyxy[1])
        return bbox_xyxy if area_xyxy <= area_xywh else bbox_xywh

    return clamp_bbox(0, 0, img_w, img_h)


def _build_background_mask_simple(
    image_bgr: np.ndarray, white_min: int, white_tolerance: int
) -> np.ndarray:
    bgr = image_bgr.astype(np.int16)
    min_c = bgr.min(axis=2)
    max_c = bgr.max(axis=2)
    return (min_c >= white_min) & ((max_c - min_c) <= white_tolerance)


def _build_background_mask_border(
    image_bgr: np.ndarray, border_percentile: float, border_margin: int
) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    white_lab = np.array([255.0, 128.0, 128.0], dtype=np.float32)
    dist = np.linalg.norm(lab - white_lab, axis=2)

    h, w = dist.shape
    margin = max(1, min(border_margin, h // 2, w // 2))
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:margin, :] = True
    border_mask[-margin:, :] = True
    border_mask[:, :margin] = True
    border_mask[:, -margin:] = True

    border_vals = dist[border_mask]
    if border_vals.size == 0:
        threshold = np.percentile(dist, border_percentile)
    else:
        threshold = np.percentile(border_vals, border_percentile)

    candidate = dist <= threshold

    labels = cv2.connectedComponents(candidate.astype(np.uint8))[1]
    border_labels = np.unique(
        np.concatenate(
            [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]
        )
    )
    bg_mask = np.isin(labels, border_labels) & candidate
    return bg_mask


def _apply_morphology(
    mask: np.ndarray, close_kernel: int, open_kernel: int
) -> np.ndarray:
    out = mask
    if close_kernel and close_kernel > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)
        )
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    if open_kernel and open_kernel > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_kernel, open_kernel)
        )
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    return out


def _relative_output_path(image_path: Path, images_dir: Path, masks_dir: Path) -> Path:
    try:
        relative = image_path.relative_to(images_dir)
    except ValueError:
        relative = Path(image_path.name)
    return masks_dir / relative.with_suffix(".png")


def _preview(image_bgr: np.ndarray, mask: np.ndarray, title: str) -> bool:
    # Resize to half for faster preview and overlay mask on the image.
    scale = 0.5
    new_size = (int(image_bgr.shape[1] * scale), int(image_bgr.shape[0] * scale))
    preview_img = cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)
    preview_mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)

    mask_bool = preview_mask > 0
    overlay = preview_img.copy()
    overlay[~mask_bool] = 0

    cv2.imshow(title, overlay)

    # Wait until the user closes the window; allow 'q' to quit early.
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            cv2.destroyWindow(title)
            return False
        visible = cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE)
        if visible < 1:
            break

    return True


def _get_sam_predictor(checkpoint: Path, device: Optional[str]):
    global _SAM_PREDICTOR
    if _SAM_PREDICTOR is not None:
        return _SAM_PREDICTOR

    try:
        import torch
        from mobile_sam import SamPredictor
        from mobile_sam.build_sam import sam_model_registry
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(f"Failed to import MobileSAM dependencies: {exc}") from exc

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")

    build_fn = sam_model_registry["vit_t"]
    model = build_fn(checkpoint=str(checkpoint))
    model.to(device)
    model.eval()

    predictor = SamPredictor(model)
    _SAM_PREDICTOR = predictor
    return predictor


def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _expand_bbox(
    bbox: Tuple[int, int, int, int], img_w: int, img_h: int, margin: int
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(img_w, x2 + margin)
    y2 = min(img_h, y2 + margin)
    return x1, y1, x2, y2


def _refine_with_sam(
    image_bgr: np.ndarray,
    init_mask: np.ndarray,
    checkpoint: Path,
    device: Optional[str],
    box_margin: int,
    multimask: bool,
) -> np.ndarray:
    predictor = _get_sam_predictor(checkpoint, device)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    mask_bbox = _mask_bbox(init_mask)
    box = None
    if mask_bbox is not None:
        img_h, img_w = init_mask.shape[:2]
        x1, y1, x2, y2 = _expand_bbox(mask_bbox, img_w, img_h, box_margin)
        box = np.array([x1, y1, x2, y2])

    mask_input_size = predictor.model.prompt_encoder.mask_input_size
    resized = cv2.resize(
        (init_mask > 0).astype(np.float32),
        (mask_input_size[1], mask_input_size[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    mask_input = resized[None, :, :]

    masks, scores, _ = predictor.predict(
        box=box, mask_input=mask_input, multimask_output=multimask
    )
    if masks.ndim != 3 or masks.shape[0] == 0:
        return init_mask

    if multimask:
        best_idx = int(np.argmax(scores))
        refined = masks[best_idx]
    else:
        refined = masks[0]

    return refined.astype(np.uint8) * 255


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root
    csv_path = args.csv or (dataset_root / DEFAULT_CSV_NAME)
    images_dir = args.images_dir or (dataset_root / DEFAULT_IMAGES_DIR_NAME)
    masks_dir = args.masks_dir or (dataset_root / DEFAULT_MASKS_DIR_NAME)
    masks_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    extensions = _parse_extensions(args.extensions)

    processed = 0
    saved = 0
    previewed = 0
    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or "imageName" not in reader.fieldnames:
            raise ValueError("CSV must contain an imageName column.")

        for row in reader:
            if args.limit is not None and processed >= args.limit:
                break
            processed += 1

            image_name = (row.get("imageName") or "").strip()
            image_path = _resolve_image_path(image_name, images_dir, extensions)
            if image_path is None:
                print(f"[WARN] Image not found for imageName='{image_name}'")
                continue

            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                print(f"[WARN] Failed to read image: {image_path}")
                continue

            img_h, img_w = image_bgr.shape[:2]
            bbox = (0, 0, img_w, img_h)
            if args.use_bbox:
                bbox_values = _parse_bbox_values((row.get("bbox") or "").strip())
                if bbox_values:
                    bbox = _infer_bbox(bbox_values, img_w, img_h, args.bbox_format)

            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                x1, y1, x2, y2 = 0, 0, img_w, img_h

            roi = image_bgr[y1:y2, x1:x2]
            if args.bg_method == "simple":
                white_mask = _build_background_mask_simple(
                    roi, args.white_min, args.white_tolerance
                )
            else:
                white_mask = _build_background_mask_border(
                    roi, args.border_percentile, args.border_margin
                )
            fg_mask = (~white_mask).astype(np.uint8) * 255
            fg_mask = _apply_morphology(fg_mask, args.close_kernel, args.open_kernel)

            full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = fg_mask

            if args.refine_sam:
                if full_mask.any():
                    try:
                        full_mask = _refine_with_sam(
                            image_bgr,
                            full_mask,
                            args.sam_checkpoint,
                            args.sam_device,
                            args.sam_box_margin,
                            args.sam_multimask,
                        )
                    except Exception as exc:
                        print(f"[WARN] SAM refinement failed for {image_path}: {exc}")
                else:
                    print(f"[WARN] Empty initial mask for {image_path}, skipping SAM.")

            output_path = _relative_output_path(image_path, images_dir, masks_dir)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), full_mask)
            saved += 1

            if args.preview and previewed < args.preview_limit:
                keep_going = _preview(image_bgr, full_mask, image_path.name)
                previewed += 1
                if not keep_going:
                    break

    print(f"Processed {processed} rows. Saved {saved} masks to {masks_dir}")


if __name__ == "__main__":
    main()
