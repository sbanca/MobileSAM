"""
Interactive MobileSAM mask generator for "PassthroughSnapshot" image sets.

Steps covered by this tool:
1. Scan a hard-coded folder and collect all PNGs containing "PassthroughSnapshot".
2. Generate initial hand masks with MobileSAM.
3. Show thumbnails (4 rows x 8 columns layout) with overlayed masks inside a PySide6 UI.
4. Let users open any thumbnail to refine the mask interactively via point clicks.
5. Save all masks with filenames starting with "Mask_" instead of "PassthroughSnapshot_Left_".

Adjust the DATASETS_ROOT constant or use --root if the dataset location changes.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PySide6.QtCore import QPoint, QSize, Qt, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from handLandmarks.handLandmarksDetection import MediaPipeTaskHandLandmarkDetector
from mobile_sam.automatic_mask_generator import SamAutomaticMaskGenerator
from mobile_sam.build_sam import sam_model_registry
from mobile_sam.predictor import SamPredictor


# Hard-coded parent folder containing subfolders with PassthroughSnapshot PNGs
DATASETS_ROOT = Path(
    r"C:\Users\Staff\OneDrive - University of Greenwich\CyberASAP\phase2\data collection\Data"
)
LIST_DISPLAY_LIMIT = 10
FILENAME_TOKEN = "PassthroughSnapshot"
REPLACEMENT_PREFIX = ("PassthroughSnapshot_Left_", "Mask_")

THUMB_COLUMNS = 8
THUMBNAIL_SIZE = QSize(200, 200)
OVERLAY_COLOR_GENERATED: Tuple[int, int, int] = (0, 255, 0)
OVERLAY_COLOR_CACHED: Tuple[int, int, int] = (40, 120, 255)
PEN_BRUSH_RADIUS = 6
EDIT_MODE_SAM = "SAM"
EDIT_MODE_FLOOD = "Flood fill"
EDIT_MODE_PEN = "Pen"
HISTORY_LIMIT = 3


@dataclass
class MaskRecord:
    image_path: Path
    image_rgb: np.ndarray
    mask: np.ndarray
    auto_mask: np.ndarray = field(repr=False)
    modified: bool = False
    from_disk: bool = False

    def reset_to_auto(self) -> None:
        self.mask = self.auto_mask.copy()
        self.modified = False

    def update_mask(self, new_mask: np.ndarray) -> bool:
        """Replace the current mask. Return True if it changed."""
        changed = not np.array_equal(self.mask, new_mask)
        if changed:
            self.mask = new_mask
            self.modified = not np.array_equal(self.mask, self.auto_mask)
        return changed

    @property
    def overlay_color(self) -> Tuple[int, int, int]:
        return OVERLAY_COLOR_CACHED if self.from_disk else OVERLAY_COLOR_GENERATED


class SamHelper:
    """Wraps model loading plus automatic and point-based mask generation."""

    def __init__(self, device: Optional[str] = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = Path("weights") / "mobile_sam.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}. Please place mobile_sam.pt under weights/."
            )

        build_fn = sam_model_registry["vit_t"]
        self.model = build_fn(checkpoint=str(checkpoint))
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.automatic_generator = SamAutomaticMaskGenerator(self.model)
        self.hand_detector = MediaPipeTaskHandLandmarkDetector()

    def generate_initial_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        mask_from_landmarks = self._generate_from_landmarks(image_rgb)
        if mask_from_landmarks is not None:
            return mask_from_landmarks
        return self._generate_from_automatic(image_rgb)

    def _generate_from_landmarks(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        points = self._detect_landmark_points(image_rgb)
        if points is None or len(points) == 0:
            return None
        mask = self._mask_from_points(image_rgb, points)
        if mask is None or not mask.any():
            return None
        return mask

    def _generate_from_automatic(self, image_rgb: np.ndarray) -> np.ndarray:
        masks = self.automatic_generator.generate(image_rgb)
        best_mask = self._select_best_mask(masks, image_rgb.shape[:2])
        if best_mask is None:
            return np.zeros(image_rgb.shape[:2], dtype=bool)
        segmentation = best_mask.get("segmentation")
        if segmentation is None:
            return np.zeros(image_rgb.shape[:2], dtype=bool)
        return segmentation.astype(bool)

    def _detect_landmark_points(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        try:
            _, landmarks_px = self.hand_detector.detect(image_bgr)
        except Exception as exc:
            warnings.warn(f"MediaPipe detection failed: {exc}", RuntimeWarning)
            return None
        if isinstance(landmarks_px, tuple):
            return None
        if getattr(landmarks_px, "size", 0) == 0:
            return None
        return landmarks_px[:, :2].astype(np.float32)

    def _mask_from_points(self, image_rgb: np.ndarray, points_xy: np.ndarray) -> Optional[np.ndarray]:
        predictor = self.create_predictor()
        predictor.set_image(image_rgb)
        point_coords = np.asarray(points_xy, dtype=np.float32)
        point_labels = np.ones(point_coords.shape[0], dtype=np.int32)
        try:
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
        except Exception as exc:
            warnings.warn(f"SAM predictor (landmark prompt) failed: {exc}", RuntimeWarning)
            return None
        if masks.size == 0:
            return None
        best_idx = int(np.argmax(scores))
        return masks[best_idx] > 0.5

    def _select_best_mask(
        self, masks: Sequence[dict], shape: Sequence[int]
    ) -> Optional[dict]:
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
            bbox = mask.get("bbox")  # XYWH
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
        return best if score(best) > -np.inf else None

    def create_predictor(self) -> SamPredictor:
        return SamPredictor(self.model)

    def mask_from_point(
        self, predictor: SamPredictor, point_xy: tuple[float, float]
    ) -> Optional[np.ndarray]:
        point_coords = np.array([[point_xy[0], point_xy[1]]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)
        try:
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
        except Exception as exc:  # pragma: no cover - GUI runtime
            print(f"[warn] SAM predictor failed: {exc}")
            return None
        if masks.size == 0:
            return None
        best_idx = int(np.argmax(scores))
        return masks[best_idx] > 0.5


class ThumbnailButton(QPushButton):
    clicked_with_index = Signal(int)

    def __init__(self, index: int, pixmap: QPixmap, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._index = index
        self.setIcon(QIcon(pixmap))
        self.setIconSize(pixmap.size())
        extra_padding = QSize(12, 12)
        self.setFixedSize(pixmap.size() + extra_padding)
        self.clicked.connect(self._handle_clicked)

    def _handle_clicked(self) -> None:
        self.clicked_with_index.emit(self._index)


class ImageClickLabel(QLabel):
    """QLabel showing an image and emitting click coordinates in image space."""

    clicked = Signal(float, float, Qt.MouseButton)
    dragged = Signal(float, float, Qt.MouseButton)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(320, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.CrossCursor)
        self._pixmap: Optional[QPixmap] = None
        self._scaled_pixmap: Optional[QPixmap] = None
        self._offset = QPoint(0, 0)
        self._scale_w = 1.0
        self._scale_h = 1.0

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self._update_scaled_pixmap()

    def resizeEvent(self, event) -> None:  # pragma: no cover - GUI runtime
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scaled_pixmap = scaled
        super().setPixmap(scaled)
        self._offset = QPoint(
            (self.width() - scaled.width()) // 2, (self.height() - scaled.height()) // 2
        )
        self._scale_w = scaled.width() / max(1, self._pixmap.width())
        self._scale_h = scaled.height() / max(1, self._pixmap.height())

    def mousePressEvent(self, event) -> None:  # pragma: no cover - GUI runtime
        if event.button() not in (Qt.LeftButton, Qt.RightButton):
            return
        mapped = self._map_position(event)
        if mapped is None:
            return
        image_x, image_y = mapped
        self.clicked.emit(image_x, image_y, event.button())

    def mouseMoveEvent(self, event) -> None:  # pragma: no cover - GUI runtime
        buttons = event.buttons()
        button = None
        if buttons & Qt.LeftButton:
            button = Qt.LeftButton
        elif buttons & Qt.RightButton:
            button = Qt.RightButton
        if button is None:
            return
        mapped = self._map_position(event)
        if mapped is None:
            return
        image_x, image_y = mapped
        self.dragged.emit(image_x, image_y, button)

    def _map_position(self, event) -> Optional[Tuple[float, float]]:
        if self._pixmap is None or self._scaled_pixmap is None:
            return None
        pos = event.position() if hasattr(event, "position") else event.pos()
        x = pos.x() - self._offset.x()
        y = pos.y() - self._offset.y()
        if not (0 <= x <= self._scaled_pixmap.width() and 0 <= y <= self._scaled_pixmap.height()):
            return None
        image_x = x / max(self._scale_w, 1e-6)
        image_y = y / max(self._scale_h, 1e-6)
        return image_x, image_y


class MaskEditorWindow(QWidget):
    """Secondary window for editing a single mask via clicks."""

    def __init__(
        self,
        record: MaskRecord,
        sam_helper: SamHelper,
        on_mask_updated: Callable[[MaskRecord], None],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.record = record
        self.sam_helper = sam_helper
        self.on_mask_updated = on_mask_updated
        self.predictor = self.sam_helper.create_predictor()
        self.predictor.set_image(self.record.image_rgb)
        self.current_mask = record.mask.copy()
        self.undo_stack: List[np.ndarray] = []
        self.redo_stack: List[np.ndarray] = []

        self.setWindowTitle(self.record.image_path.name)
        self.resize(1000, 600)

        self.original_label = ImageClickLabel()
        self.mask_label = ImageClickLabel()

        info_label = QLabel(
            "SAM: left click adds segments, right click removes.\n"
            "Flood fill: left click fills black, right click fills white.\n"
            "Pen: left click draws white, right click draws black."
        )
        self.mode_selector = QComboBox()
        self.mode_selector.addItems([EDIT_MODE_SAM, EDIT_MODE_FLOOD, EDIT_MODE_PEN])
        self.mode_selector.setToolTip("Select how clicks modify the mask.")

        undo_btn = QPushButton("Undo")
        redo_btn = QPushButton("Redo")
        self.undo_button = undo_btn
        self.redo_button = redo_btn
        grow_btn = QPushButton("Grow 1 px")
        shrink_btn = QPushButton("Shrink 1 px")
        reset_btn = QPushButton("Reset to auto mask")
        close_btn = QPushButton("Close editor")

        undo_btn.setToolTip("Undo last mask change (up to 3).")
        redo_btn.setToolTip("Redo last undone change (up to 3).")
        grow_btn.setToolTip("Expand white mask regions by 1 pixel.")
        shrink_btn.setToolTip("Contract white mask regions by 1 pixel.")
        reset_btn.clicked.connect(self._handle_reset)
        undo_btn.clicked.connect(self._handle_undo)
        redo_btn.clicked.connect(self._handle_redo)
        grow_btn.clicked.connect(self._handle_grow)
        shrink_btn.clicked.connect(self._handle_shrink)
        close_btn.clicked.connect(self.close)
        self.original_label.clicked.connect(self._handle_click)
        self.mask_label.clicked.connect(self._handle_click)
        self.original_label.dragged.connect(self._handle_drag)
        self.mask_label.dragged.connect(self._handle_drag)

        button_layout = QHBoxLayout()
        button_layout.addWidget(undo_btn)
        button_layout.addWidget(redo_btn)
        button_layout.addWidget(grow_btn)
        button_layout.addWidget(shrink_btn)
        button_layout.addWidget(reset_btn)
        button_layout.addWidget(close_btn)
        button_layout.addStretch()

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Edit mode:"))
        mode_layout.addWidget(self.mode_selector)
        mode_layout.addStretch()

        images_layout = QHBoxLayout()
        images_layout.addWidget(self.original_label)
        images_layout.addWidget(self.mask_label)
        images_layout.setStretch(0, 1)
        images_layout.setStretch(1, 1)

        main_layout = QVBoxLayout()
        main_layout.addWidget(info_label)
        main_layout.addLayout(mode_layout)
        main_layout.addLayout(images_layout)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)
        self._refresh_views()
        self._update_history_buttons()

    def _handle_click(self, x: float, y: float, button: Qt.MouseButton) -> None:
        mode = self.mode_selector.currentText()
        if mode == EDIT_MODE_FLOOD:
            fill_white = button == Qt.RightButton
            self._apply_new_mask(flood_fill_mask(self.current_mask, (x, y), fill_white))
            return
        if mode == EDIT_MODE_PEN:
            draw_white = button == Qt.LeftButton
            self._apply_new_mask(draw_on_mask(self.current_mask, (x, y), PEN_BRUSH_RADIUS, draw_white))
            return
        mask = self.sam_helper.mask_from_point(self.predictor, (x, y))
        if mask is None:
            return
        mask_bool = mask.astype(bool)
        if button == Qt.RightButton:
            new_mask = np.logical_and(self.current_mask, ~mask_bool)
        else:
            new_mask = np.logical_or(self.current_mask, mask_bool)
        self._apply_new_mask(new_mask)

    def _handle_drag(self, x: float, y: float, button: Qt.MouseButton) -> None:
        if self.mode_selector.currentText() != EDIT_MODE_PEN:
            return
        draw_white = button == Qt.LeftButton
        self._apply_new_mask(draw_on_mask(self.current_mask, (x, y), PEN_BRUSH_RADIUS, draw_white))

    def _apply_new_mask(self, new_mask: np.ndarray, *, record_history: bool = True) -> None:
        changed = not np.array_equal(self.current_mask, new_mask)
        if record_history and changed:
            self.undo_stack.append(self.current_mask.copy())
            if len(self.undo_stack) > HISTORY_LIMIT:
                self.undo_stack.pop(0)
            self.redo_stack.clear()
        self.current_mask = new_mask
        mask_changed = self.record.update_mask(new_mask)
        self._refresh_views()
        if mask_changed:
            self.on_mask_updated(self.record)
        self._update_history_buttons()

    def _handle_undo(self) -> None:
        if not self.undo_stack:
            return
        self.redo_stack.append(self.current_mask.copy())
        if len(self.redo_stack) > HISTORY_LIMIT:
            self.redo_stack.pop(0)
        previous = self.undo_stack.pop()
        self._apply_new_mask(previous.copy(), record_history=False)

    def _handle_redo(self) -> None:
        if not self.redo_stack:
            return
        self.undo_stack.append(self.current_mask.copy())
        if len(self.undo_stack) > HISTORY_LIMIT:
            self.undo_stack.pop(0)
        restored = self.redo_stack.pop()
        self._apply_new_mask(restored.copy(), record_history=False)

    def _update_history_buttons(self) -> None:
        self.undo_button.setEnabled(bool(self.undo_stack))
        self.redo_button.setEnabled(bool(self.redo_stack))

    def _handle_reset(self) -> None:
        self._apply_new_mask(self.record.auto_mask.copy())

    def _handle_grow(self) -> None:
        self._apply_new_mask(grow_mask_one_pixel(self.current_mask))

    def _handle_shrink(self) -> None:
        self._apply_new_mask(shrink_mask_one_pixel(self.current_mask))

    def _refresh_views(self) -> None:
        overlay = apply_mask_overlay(
            self.record.image_rgb,
            self.current_mask,
            color=self.record.overlay_color,
        )
        overlay_pix = numpy_to_qpixmap(overlay)
        self.original_label.set_pixmap(overlay_pix)

        mask_rgb = mask_to_rgb(self.current_mask)
        mask_pix = numpy_to_qpixmap(mask_rgb)
        self.mask_label.set_pixmap(mask_pix)


class ThumbnailGridWindow(QWidget):
    """Main window showing thumbnails and Save All button."""

    def __init__(self, records: Sequence[MaskRecord], sam_helper: SamHelper, dataset_folder: Path) -> None:
        super().__init__()
        self.records = list(records)
        self.sam_helper = sam_helper
        self.editors: List[MaskEditorWindow] = []
        self.setWindowTitle(f"PassthroughSnapshot Mask Review — {dataset_folder.name}")
        self.resize(1400, 900)

        header_label = QLabel(dataset_folder.name)
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setStyleSheet("font-size: 22pt; font-weight: 700; padding: 8px 0;")

        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(8)
        self.thumbnail_buttons: List[ThumbnailButton] = []
        for idx, record in enumerate(self.records):
            pixmap = build_thumbnail_pixmap(record, THUMBNAIL_SIZE)
            button = ThumbnailButton(idx, pixmap)
            button.setToolTip(record.image_path.name)
            button.clicked_with_index.connect(self._open_editor)
            row = idx // THUMB_COLUMNS
            col = idx % THUMB_COLUMNS
            self.grid_layout.addWidget(button, row, col)
            self.thumbnail_buttons.append(button)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.grid_widget)

        save_btn = QPushButton("Save All Masks")
        save_btn.setFixedHeight(40)
        save_btn.clicked.connect(self._save_all_masks)

        main_layout = QVBoxLayout()
        main_layout.addWidget(header_label)
        main_layout.addWidget(scroll_area)
        main_layout.addWidget(save_btn)
        self.setLayout(main_layout)

    def _open_editor(self, index: int) -> None:
        record = self.records[index]
        editor = MaskEditorWindow(record, self.sam_helper, self._handle_mask_update)
        editor.setAttribute(Qt.WA_DeleteOnClose)
        editor.show()
        self.editors.append(editor)

    def _handle_mask_update(self, record: MaskRecord) -> None:
        index = self.records.index(record)
        pixmap = build_thumbnail_pixmap(record, THUMBNAIL_SIZE)
        button = self.thumbnail_buttons[index]
        button.setIcon(QIcon(pixmap))
        button.setIconSize(pixmap.size())

    def _save_all_masks(self) -> None:  # pragma: no cover - GUI runtime
        failures: List[str] = []
        saved_paths: List[Path] = []
        for record in self.records:
            output_path = build_output_path(record.image_path)
            try:
                save_mask_to_disk(record.mask, output_path)
                saved_paths.append(output_path)
            except Exception as exc:
                failures.append(f"{record.image_path.name}: {exc}")
        if failures:
            QMessageBox.warning(
                self,
                "Save masks",
                "Some masks failed to save:\n" + "\n".join(failures),
            )
        else:
            QMessageBox.information(
                self,
                "Save masks",
                f"Saved {len(saved_paths)} masks.\nExample: {saved_paths[0]}" if saved_paths else "No masks saved.",
            )


def apply_mask_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    color: Tuple[int, int, int] = OVERLAY_COLOR_GENERATED,
    alpha: float = 0.45,
) -> np.ndarray:
    overlay = image_rgb.copy()
    mask_bool = mask.astype(bool)
    color_arr = np.array(color, dtype=np.float32)
    blended = overlay.astype(np.float32)
    blended[mask_bool] = (
        (1.0 - alpha) * blended[mask_bool] + alpha * color_arr
    )
    return blended.clip(0, 255).astype(np.uint8)


def grow_mask_one_pixel(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return mask
    mask_u8 = np.where(mask, 255, 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    grown = cv2.dilate(mask_u8, kernel, iterations=1)
    return grown > 127


def shrink_mask_one_pixel(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return mask
    mask_u8 = np.where(mask, 255, 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    shrunk = cv2.erode(mask_u8, kernel, iterations=1)
    return shrunk > 127


def draw_on_mask(
    mask: np.ndarray, center_xy: tuple[float, float], radius: int, draw_white: bool
) -> np.ndarray:
    height, width = mask.shape[:2]
    if height == 0 or width == 0:
        return mask
    cx = int(round(center_xy[0]))
    cy = int(round(center_xy[1]))
    if cx < 0 or cx >= width or cy < 0 or cy >= height:
        return mask

    y0 = max(0, cy - radius)
    y1 = min(height, cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(width, cx + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return mask

    yy, xx = np.ogrid[y0:y1, x0:x1]
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    updated = mask.copy()
    if draw_white:
        updated[y0:y1, x0:x1][circle] = True
    else:
        updated[y0:y1, x0:x1][circle] = False
    return updated


def flood_fill_mask(
    mask: np.ndarray, seed_xy: tuple[float, float], fill_white: bool
) -> np.ndarray:
    height, width = mask.shape[:2]
    if height == 0 or width == 0:
        return mask
    x = int(round(seed_xy[0]))
    y = int(round(seed_xy[1]))
    if x < 0 or x >= width or y < 0 or y >= height:
        return mask

    mask_u8 = np.ascontiguousarray(np.where(mask, 255, 0).astype(np.uint8))
    new_val = 255 if fill_white else 0
    if int(mask_u8[y, x]) == new_val:
        return mask

    fill_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    flags = 4 | cv2.FLOODFILL_FIXED_RANGE
    cv2.floodFill(
        mask_u8,
        fill_mask,
        seedPoint=(x, y),
        newVal=new_val,
        loDiff=0,
        upDiff=0,
        flags=flags,
    )
    return mask_u8 > 127


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    mask_u8 = np.where(mask, 255, 0).astype(np.uint8)
    return np.repeat(mask_u8[..., None], 3, axis=2)


def numpy_to_qpixmap(image: np.ndarray) -> QPixmap:
    if image.ndim == 2:
        arr = np.ascontiguousarray(image)
        height, width = arr.shape
        qimage = QImage(arr.data, width, height, arr.strides[0], QImage.Format_Grayscale8)
    else:
        arr = np.ascontiguousarray(image)
        height, width, _ = arr.shape
        qimage = QImage(arr.data, width, height, arr.strides[0], QImage.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


def build_thumbnail_pixmap(record: MaskRecord, target_size: QSize) -> QPixmap:
    overlay = apply_mask_overlay(record.image_rgb, record.mask, color=record.overlay_color)
    pixmap = numpy_to_qpixmap(overlay)
    return pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def build_output_path(image_path: Path) -> Path:
    before, after = REPLACEMENT_PREFIX
    name = image_path.name
    if before in name:
        name = name.replace(before, after, 1)
    else:
        name = after + name
    return image_path.with_name(name)


def save_mask_to_disk(mask: np.ndarray, output_path: Path) -> None:
    mask_u8 = np.where(mask, 255, 0).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), mask_u8):
        raise RuntimeError("cv2.imwrite returned False")


def load_image_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def find_passthrough_images(root: Path, *, raise_if_missing: bool = True) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input folder does not exist: {root}")
    matches = [
        p
        for p in sorted(root.rglob("*.png"))
        if FILENAME_TOKEN in p.name
    ]
    if not matches:
        if raise_if_missing:
            raise RuntimeError(f"No PNG files containing '{FILENAME_TOKEN}' found under {root}")
        return []
    return matches


def load_existing_mask(mask_path: Path) -> Optional[np.ndarray]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        warnings.warn(f"Failed to read cached mask: {mask_path}", RuntimeWarning)
        return None
    return mask > 127


def list_folders_missing_masks(root: Path) -> List[Path]:
    missing: List[Path] = []
    if not root.exists():
        return missing
    subfolders = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    for folder in subfolders:
        images = find_passthrough_images(folder, raise_if_missing=False)
        if not images:
            continue
        needs_masks = any(not build_output_path(img).exists() for img in images)
        if needs_masks:
            missing.append(folder)
    return missing


def summarize_folder_coverage(root: Path) -> Tuple[int, int, int]:
    total = 0
    with_masks = 0
    without_masks = 0
    subfolders = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    for folder in subfolders:
        images = find_passthrough_images(folder, raise_if_missing=False)
        if not images:
            continue
        total += 1
        needs_masks = any(not build_output_path(img).exists() for img in images)
        if needs_masks:
            without_masks += 1
        else:
            with_masks += 1
    return total, with_masks, without_masks


def prompt_for_dataset_folder(
    root: Path,
    missing_folders: Sequence[Path],
    limit: int = LIST_DISPLAY_LIMIT,
) -> Path:
    display_count = min(limit, len(missing_folders))
    if missing_folders:
        print(f"\nFolders missing masks (showing up to {display_count}):")
        for idx, folder in enumerate(missing_folders[:display_count], start=1):
            print(f"  {idx}. {folder.name}")
        print(
            "\nEnter a number from the list above to open it, "
            "or type a folder name/path (relative to root) to open another folder."
        )
        print("Press Enter with no input to open the first folder in the list.")
    else:
        print(
            "\nAll subfolders appear to contain masks. "
            "Type the name of a folder to open it (relative to root) or an absolute path."
        )

    while True:
        choice = input("Folder selection: ").strip()
        if not choice:
            if missing_folders:
                return missing_folders[0]
            print("Please enter a folder name or absolute path.")
            continue
        if choice.isdigit() and display_count > 0:
            idx = int(choice)
            if 1 <= idx <= display_count:
                return missing_folders[idx - 1]
            print(f"Enter a number between 1 and {display_count}.")
            continue
        candidate = Path(choice)
        if not candidate.is_absolute():
            candidate = root / choice
        if candidate.is_dir():
            return candidate
        print(f"Folder not found: {candidate}. Try again.")


def build_records(helper: SamHelper, image_paths: Sequence[Path]) -> List[MaskRecord]:
    records: List[MaskRecord] = []
    total = len(image_paths)
    for idx, image_path in enumerate(image_paths, start=1):
        image_rgb = load_image_rgb(image_path)
        mask: Optional[np.ndarray] = None
        from_disk = False

        cached_mask_path = build_output_path(image_path)
        if cached_mask_path.exists():
            mask = load_existing_mask(cached_mask_path)
            if mask is not None:
                from_disk = True
                print(f"[{idx}/{total}] Loaded cached mask {cached_mask_path.name}")

        if mask is None:
            print(f"[{idx}/{total}] Generating mask for {image_path.name}")
            mask = helper.generate_initial_mask(image_rgb)

        record = MaskRecord(
            image_path=image_path,
            image_rgb=image_rgb,
            mask=mask.copy(),
            auto_mask=mask.copy(),
            from_disk=from_disk,
        )
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive MobileSAM labeling assistant.")
    parser.add_argument(
        "--root",
        type=str,
        default=str(DATASETS_ROOT),
        help="Parent folder containing per-session subfolders (default: %(default)s).",
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Specific subfolder to open (absolute path or relative to --root).",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    if not root.is_dir():
        raise SystemExit(f"Root folder not found: {root}")

    if args.folder:
        candidate = Path(args.folder).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        if not candidate.is_dir():
            raise SystemExit(f"Dataset folder not found: {candidate}")
        target_folder = candidate
    else:
        missing = list_folders_missing_masks(root)
        total, with_masks, without_masks = summarize_folder_coverage(root)
        if total > 0:
            pct_with = (with_masks / total) * 100.0
            pct_without = (without_masks / total) * 100.0
            print(
                f"\nFolder coverage summary: {with_masks}/{total} ({pct_with:.1f}%) with masks, "
                f"{without_masks}/{total} ({pct_without:.1f}%) without masks."
            )
        target_folder = prompt_for_dataset_folder(root, missing, LIST_DISPLAY_LIMIT)

    print(f"\nScanning for PassthroughSnapshot PNG files under: {target_folder}")
    image_paths = find_passthrough_images(target_folder)
    print(f"Found {len(image_paths)} candidates.")
    sam_helper = SamHelper()
    records = build_records(sam_helper, image_paths)

    app = QApplication(sys.argv)
    window = ThumbnailGridWindow(records, sam_helper, target_folder)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
