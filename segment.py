import os
import traceback
from typing import List, Dict, Any, Optional, Sequence, Tuple

import numpy as np
import torch

from mobile_sam.build_sam import sam_model_registry
from mobile_sam.automatic_mask_generator import SamAutomaticMaskGenerator

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # OpenCV is optional for this script


def load_image_rgb(path: str) -> np.ndarray:
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return np.array(img)
    except Exception as e:
        raise RuntimeError(f"Failed to load image '{path}': {e}")


def load_depth_mask(path: str) -> np.ndarray:
    """Load a depth mask image and return a boolean mask where white pixels are True."""
    try:
        from PIL import Image
        img = Image.open(path).convert("L")
        arr = np.array(img)
        return arr > 127
    except Exception as e:
        raise RuntimeError(f"Failed to load depth mask '{path}': {e}")


def save_centroids_debug_image(
    depth_mask: np.ndarray,
    centroids: List[Tuple[float, float]],
    out_path: str,
) -> None:
    """Save an RGB debug image of the depth mask with all centroids drawn.

    - depth_mask: boolean array shape (H, W)
    - centroids: list of (y, x) float coordinates
    - out_path: path to save PNG
    """
    from PIL import Image, ImageDraw

    # Create RGB image: white where True, black where False
    h, w = depth_mask.shape[:2]
    base = np.zeros((h, w, 3), dtype=np.uint8)
    base[depth_mask] = 255

    img = Image.fromarray(base, mode="RGB")
    draw = ImageDraw.Draw(img)

    # Draw small circles, color-coded if they fall on white
    r = 3
    for (yy, xx) in centroids:
        y = int(round(yy))
        x = int(round(xx))
        color = (0, 200, 0) if (0 <= y < h and 0 <= x < w and depth_mask[y, x]) else (220, 0, 0)
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)

    img.save(out_path)


def combine_masks(masks: List[Dict[str, Any]], shape: tuple) -> np.ndarray:
    combined = np.zeros(shape[:2], dtype=np.uint8)
    for m in masks:
        seg = m.get("segmentation")
        if seg is None:
            # Some output modes may not include 'segmentation'
            continue
        combined |= seg.astype(np.uint8)
    return combined * 255  # scale to 0/255 for saving as image


def parse_bbox(value: str) -> Optional[Tuple[int, int, int, int]]:
    value = value.strip()
    if not value:
        return None
    try:
        parts = [int(float(token)) for token in value.split(",")]
    except ValueError:
        return None
    if len(parts) != 4:
        return None
    xmin, ymin, xmax, ymax = parts
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def mask_center(segmentation: np.ndarray) -> Optional[Tuple[float, float]]:
    coords = np.argwhere(segmentation)
    if coords.size == 0:
        return None
    yx = coords.mean(axis=0).astype(np.float32)
    return float(yx[0]), float(yx[1])


def segment_select_from_rgb_and_depth(
    image_rgb: np.ndarray,
    depth_mask_img: np.ndarray,
    *,
    model: Optional[torch.nn.Module] = None,
    device: Optional[str] = None,
    overlap_ratio_threshold: float = 0.75,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[Tuple[float, float]]]:
    """
    Segment the RGB image with MobileSAM and select segments based on the depth mask.

    Criteria:
    - Keep segments with overlap ratio > overlap_ratio_threshold
      (fraction of segment pixels that are white in depth mask). By default,
      this means more than half of the segment’s pixels must be white.

    Returns:
      combined_mask_uint8, kept_masks, kept_centroids
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Normalize depth mask to boolean
    if depth_mask_img.dtype != np.bool_:
        if depth_mask_img.ndim == 3:
            depth_gray = depth_mask_img[..., 0]
        else:
            depth_gray = depth_mask_img
        depth_mask_bool = depth_gray > 127
    else:
        depth_mask_bool = depth_mask_img

    if image_rgb.shape[:2] != depth_mask_bool.shape[:2]:
        raise ValueError(
            f"Image shape {image_rgb.shape[:2]} and depth mask shape {depth_mask_bool.shape[:2]} must match."
        )

    # Build model if not supplied
    if model is None:
        checkpoint_path = os.path.join("weights", "mobile_sam.pt")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. Expected weights/mobile_sam.pt"
            )
        build_fn = sam_model_registry["vit_t"]
        model = build_fn(checkpoint=checkpoint_path)
        model.to(device)
        model.eval()

    mask_generator = SamAutomaticMaskGenerator(model)
    masks: List[Dict[str, Any]] = mask_generator.generate(image_rgb)

    # Filter by overlap threshold only (no centroid requirement)
    kept_masks: List[Dict[str, Any]] = []
    for m in masks:
        seg = m.get("segmentation")
        if seg is None:
            continue
        seg_bool = seg.astype(bool)
        seg_area = int(seg_bool.sum())
        if seg_area == 0:
            continue
        overlap = int((seg_bool & depth_mask_bool).sum())
        ratio = overlap / max(1, seg_area)
        if ratio > overlap_ratio_threshold:
            kept_masks.append(m)
            
    # Combine kept masks
    combined_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    for m in kept_masks:
        seg = m.get("segmentation")
        if seg is None:
            continue
        combined_mask |= seg.astype(np.uint8)
    combined_mask *= 255

    return combined_mask, kept_masks


def save_png(path: str, array: np.ndarray) -> None:
    try:
        from PIL import Image
        mode = "L" if array.ndim == 2 else "RGB"
        Image.fromarray(array, mode=mode).save(path)
    except Exception as e:
        raise RuntimeError(f"Failed to save image '{path}': {e}")


def main() -> None:
    # Hardcoded inputs
    image_path = "10_13575_left_dorsal_rgb.png"
    depth_mask_path = "10_13575_left_dorsal_mask.png"
    checkpoint_path = os.path.join("weights", "mobile_sam.pt")

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Expected weights/mobile_sam.pt"
        )

    # Load image as RGB numpy array (H, W, 3), dtype=uint8
    image = load_image_rgb(image_path)
    depth_mask = load_depth_mask(depth_mask_path)
    combined_mask, kept_masks, kept_centroids = segment_select_from_rgb_and_depth(
        image, depth_mask
    )

    # Save outputs
    base, _ = os.path.splitext(os.path.basename(image_path))
    debug_out = f"{base}_depth_centroids.png"
    save_centroids_debug_image(depth_mask, kept_centroids, debug_out)
    print(f"Saved centroids debug image to: {debug_out}")

    combined_out = f"{base}_kept_combined_mask.png"
    save_png(combined_out, combined_mask)
    print(f"Saved kept combined mask to: {combined_out}")

    # Save overlay with the kept combined mask
    overlay = image.copy()
    red = np.zeros_like(image)
    red[..., 0] = 255
    alpha = 0.5
    mask_bool = combined_mask > 0
    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool] + alpha * red[mask_bool]
    ).astype(np.uint8)
    overlay_out = f"{base}_kept_overlay.png"
    save_png(overlay_out, overlay)
    print(f"Saved kept overlay to: {overlay_out}")


if __name__ == "__main__":
    main()
