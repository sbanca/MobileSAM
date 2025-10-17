import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple
import warnings
from numbers import Real

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import cv2

from load_hand_dataset import load_dataset
from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry


def parse_bbox(value: str) -> Optional[Tuple[int, int, int, int]]:
    """Parse bbox string 'xmin,ymin,xmax,ymax' into integers."""
    value = value.strip()
    if not value:
        return None

    try:
        parts = [int(float(token)) for token in value.split(",")]
    except ValueError:
        warnings.warn(f"Invalid bbox format: '{value}'", RuntimeWarning)
        return None

    if len(parts) != 4:
        warnings.warn(f"Bbox does not have 4 values: '{value}'", RuntimeWarning)
        return None

    xmin, ymin, xmax, ymax = parts
    if xmax <= xmin or ymax <= ymin:
        warnings.warn(f"Bbox has non-positive area: '{value}'", RuntimeWarning)
        return None

    return xmin, ymin, xmax, ymax


def bbox_center(bbox: Sequence[int]) -> np.ndarray:
    """Return (y, x) coordinates of the bbox barycenter."""
    xmin, ymin, xmax, ymax = bbox
    return np.array([(ymin + ymax) / 2.0, (xmin + xmax) / 2.0], dtype=np.float32)


def mask_center(mask: np.ndarray) -> Optional[np.ndarray]:
    """Return (y, x) barycenter of the mask."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return coords.mean(axis=0).astype(np.float32)


def select_mask_by_bbox(
    masks: Sequence[dict], bbox: Sequence[int]
) -> Optional[dict]:
    """Choose the mask with the highest overlap relative to the bbox."""
    if not masks:
        return None

    best_mask_dict: Optional[dict] = None
    best_overlap = -np.inf

    for mask in masks:
        segmentation = mask.get("segmentation")
        if segmentation is None:
            continue
        height, width = segmentation.shape
        xmin, ymin, xmax, ymax = bbox
        xmin_clamped = max(0, min(width - 1, xmin))
        xmax_clamped = max(0, min(width - 1, xmax))
        ymin_clamped = max(0, min(height - 1, ymin))
        ymax_clamped = max(0, min(height - 1, ymax))

        if xmax_clamped < xmin_clamped or ymax_clamped < ymin_clamped:
            overlap = 0
        else:
            overlap = segmentation[
                ymin_clamped : ymax_clamped + 1, xmin_clamped : xmax_clamped + 1
            ].sum()

        if overlap > best_overlap:
            best_overlap = overlap
            best_mask_dict = mask

    return best_mask_dict


def show_masked_image(
    image: np.ndarray,
    best_mask: dict,
    bbox: Sequence[int],
    all_masks: Sequence[dict],
    title: str,
) -> None:
    """Display the original image and masked hand with bbox and overlays."""
    best_segmentation = best_mask.get("segmentation")
    if best_segmentation is None:
        warnings.warn("Best mask missing segmentation data; skipping display.", RuntimeWarning)
        return

    masked = image.copy()
    masked[~best_segmentation] = 0

    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title)

    ax_orig, ax_masked = axes

    ax_orig.set_title("Original + Masks")
    ax_orig.imshow(image)
    ax_orig.axis("off")

    cmap = plt.get_cmap("tab20")
    for idx, mask_dict in enumerate(all_masks):
        mask_segmentation = mask_dict.get("segmentation")
        if mask_segmentation is None:
            continue
        color = cmap(idx % 20)
        overlay = np.zeros((*mask_segmentation.shape, 4), dtype=np.float32)
        overlay[..., :3] = color[:3]
        overlay[..., 3] = mask_segmentation.astype(np.float32) * 0.3
        ax_orig.imshow(overlay)
        center = mask_center(mask_segmentation)
        if center is not None:
            ax_orig.scatter(
                center[1],
                center[0],
                s=40,
                c=[color[:3]],
                edgecolors="black",
                linewidths=0.5,
            )
            weight = mask_dict.get("predicted_iou", mask_dict.get("area"))
            if weight is not None:
                ax_orig.text(
                    center[1],
                    center[0],
                    f"{weight:.2f}" if isinstance(weight, Real) else str(weight),
                    color="white",
                    fontsize=7,
                    ha="center",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.6),
                )

    rect_orig = Rectangle(
        (xmin, ymin),
        width,
        height,
        linewidth=2,
        edgecolor="lime",
        facecolor="none",
    )
    ax_orig.add_patch(rect_orig)

    ax_masked.set_title("Masked Hand")
    ax_masked.imshow(masked)
    ax_masked.axis("off")
    rect_masked = Rectangle(
        (xmin, ymin),
        width,
        height,
        linewidth=2,
        edgecolor="lime",
        facecolor="none",
    )
    ax_masked.add_patch(rect_masked)
    best_center = mask_center(best_segmentation)
    if best_center is not None:
        ax_masked.scatter(
            best_center[1],
            best_center[0],
            s=60,
            c=["yellow"],
            edgecolors="black",
            linewidths=0.6,
        )
        weight = best_mask.get("predicted_iou", best_mask.get("area"))
        if weight is not None:
            ax_masked.text(
                best_center[1],
                best_center[0],
                f"{weight:.2f}" if isinstance(weight, Real) else str(weight),
                color="black",
                fontsize=8,
                ha="center",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="yellow", ec="black", alpha=0.7),
            )

    fig.tight_layout()
    plt.show()


def save_mask(segmentation: np.ndarray, image_path: Path) -> Path:
    """Save the binary mask as an 8-bit PNG in the rgb_hand_mask folder."""
    output_dir = image_path.parent.parent / "rgb_hand_mask"
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_uint8 = segmentation.astype(np.uint8) * 255
    output_path = output_dir / image_path.name
    cv2.imwrite(str(output_path), mask_uint8)
    return output_path


def main(show_visualization: bool = False) -> None:
    dataset = load_dataset(update_csv=False)

    model_type = "vit_t"
    checkpoint = "weights/mobile_sam.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    for metadata, image_path, image_rgb in dataset:
        bbox = parse_bbox(metadata.get("bbox", ""))
        if bbox is None:
            warnings.warn(f"Skipping {image_path}: missing or invalid bbox", RuntimeWarning)
            continue

        masks = mask_generator.generate(image_rgb)
        best_mask_dict = select_mask_by_bbox(masks, bbox)
        if best_mask_dict is None:
            warnings.warn(f"No suitable mask found for {image_path}", RuntimeWarning)
            continue

        best_segmentation = best_mask_dict.get("segmentation")
        if best_segmentation is None:
            warnings.warn(f"Best mask missing segmentation for {image_path}", RuntimeWarning)
            continue

        saved_path = save_mask(best_segmentation, image_path)
        print(f"Saved mask to {saved_path}")

        if show_visualization:
            show_masked_image(image_rgb, best_mask_dict, bbox, masks, Path(image_path).name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save SAM masks for hand images.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display visualization for each image.",
    )
    args = parser.parse_args()
    main(show_visualization=args.show)
