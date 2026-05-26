"""
Interactive MobileSAM mask reviewer for the KE4ImpactFund Prolific dataset.

This follows the same workflow as passthrough_sam_gui.py, but it is tailored to:
1. Read participant image folders under prolific_dataset/participants.
2. Seed initial masks from prolific_image_landmarks.csv when landmark rows exist.
3. Open a thumbnail grid so each image can be refined with SAM clicks, flood fill,
   or pen edits.
4. Save masks under each participant folder by default:
   participants/<prolific_pid>/masks/Mask_<image_stem>.png
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import onnxruntime as ort

from PySide6.QtCore import QPoint, QSize, Qt, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
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

from mobile_sam.automatic_mask_generator import SamAutomaticMaskGenerator
from mobile_sam.build_sam import sam_model_registry
from mobile_sam.predictor import SamPredictor


DEFAULT_DATASET_ROOT = Path(
    r"C:\Users\Staff\OneDrive - University of Greenwich\KE4ImpactFund\prolific_dataset"
)
DEFAULT_CSV_NAME = "prolific_image_landmarks.csv"
DEFAULT_MASK_LANDMARK_ONNX = Path(__file__).resolve().parent / "models" / "onnx" / "mask_landmark.onnx"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LIST_DISPLAY_LIMIT = 10
THUMB_COLUMNS = 8
THUMBNAIL_SIZE = QSize(200, 200)
OVERLAY_COLOR_GENERATED: Tuple[int, int, int] = (0, 255, 0)
OVERLAY_COLOR_CACHED: Tuple[int, int, int] = (40, 120, 255)
PEN_BRUSH_RADIUS = 6
EDIT_MODE_SAM = "SAM"
EDIT_MODE_FLOOD = "Flood fill"
EDIT_MODE_PEN = "Pen"
EDIT_MODE_LANDMARKS = "Landmarks"
HISTORY_LIMIT = 3
LANDMARK_RADIUS = 8
LANDMARK_HIT_RADIUS = 28
AUTOMATIC_MAX_SIDE = 1024
DEFAULT_HAND_TEMPLATE_SCALE = 0.24
MASK_LANDMARK_INPUT_SIZE = 224
MASK_LANDMARK_MARGIN_RATIO = 0.12
MASK_SUBDIR = "masks"
MASK_PREFIX = "Mask_"
MASK_EXTENSION = ".png"


@dataclass
class MaskRecord:
    image_path: Path
    image_rgb: np.ndarray
    mask: np.ndarray
    auto_mask: np.ndarray = field(repr=False)
    landmarks_px: Optional[np.ndarray] = None
    landmarks_px_xyz: Optional[np.ndarray] = None
    landmarks_originally_missing: bool = False
    landmarks_source: str = ""
    landmarks_dirty: bool = False
    modified: bool = False
    from_disk: bool = False

    def update_mask(self, new_mask: np.ndarray) -> bool:
        changed = not np.array_equal(self.mask, new_mask)
        if changed:
            self.mask = new_mask
            self.modified = not np.array_equal(self.mask, self.auto_mask)
        return changed

    def update_landmarks(self, landmarks_px: np.ndarray) -> None:
        self.landmarks_px = landmarks_px.astype(np.float32).copy()
        if self.landmarks_px_xyz is not None and self.landmarks_px_xyz.shape[0] == self.landmarks_px.shape[0]:
            self.landmarks_px_xyz = self.landmarks_px_xyz.copy()
            self.landmarks_px_xyz[:, :2] = self.landmarks_px
        self.landmarks_dirty = True

    @property
    def overlay_color(self) -> Tuple[int, int, int]:
        return OVERLAY_COLOR_CACHED if self.from_disk else OVERLAY_COLOR_GENERATED


class OnnxMaskLandmarkDetector:
    """Predict 21 hand landmarks from a binary hand mask."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MASK_LANDMARK_ONNX,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        self.model_path = model_path
        if not self.model_path.exists():
            raise FileNotFoundError(f"Mask landmark ONNX model not found: {self.model_path}")
        preferred_providers = list(providers) if providers is not None else [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        available = set(ort.get_available_providers())
        selected = [provider for provider in preferred_providers if provider in available]
        if not selected:
            selected = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.model_path), providers=selected)
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        shape = input_info.shape
        batch_dim = shape[0] if shape else None
        self.fixed_batch = batch_dim if isinstance(batch_dim, int) and batch_dim > 0 else None

    def detect(self, mask: np.ndarray) -> Optional[np.ndarray]:
        mask_bool = np.asarray(mask).astype(bool)
        if mask_bool.ndim != 2 or not mask_bool.any():
            return None

        processed, square_origin, square_side = self._crop_square_to_model_input(mask_bool)
        batch = processed[None, None, :, :].astype(np.float32)
        if self.fixed_batch and self.fixed_batch > 1:
            batch = np.repeat(batch, self.fixed_batch, axis=0)

        output = self.session.run(None, {self.input_name: batch})[0]
        heatmaps = output[0]
        coords_model = self._heatmaps_to_coords(heatmaps)
        coords = coords_model.astype(np.float32)
        coords[:, 0] = square_origin[0] + coords[:, 0] * (square_side / MASK_LANDMARK_INPUT_SIZE)
        coords[:, 1] = square_origin[1] + coords[:, 1] * (square_side / MASK_LANDMARK_INPUT_SIZE)

        height, width = mask_bool.shape[:2]
        coords[:, 0] = np.clip(coords[:, 0], 0, max(width - 1, 0))
        coords[:, 1] = np.clip(coords[:, 1], 0, max(height - 1, 0))
        return coords

    @staticmethod
    def _heatmaps_to_coords(heatmaps: np.ndarray) -> np.ndarray:
        if heatmaps.ndim != 3:
            raise ValueError(f"Expected heatmaps with shape (21,H,W), got {heatmaps.shape}")
        landmarks, _, width = heatmaps.shape
        flat = heatmaps.reshape(landmarks, -1)
        indices = flat.argmax(axis=1)
        ys = indices // width
        xs = indices % width
        return np.stack([xs, ys], axis=1).astype(np.float32)

    @staticmethod
    def _crop_square_to_model_input(mask_bool: np.ndarray) -> tuple[np.ndarray, tuple[float, float], float]:
        height, width = mask_bool.shape[:2]
        ys, xs = np.nonzero(mask_bool)
        x1 = float(xs.min())
        x2 = float(xs.max() + 1)
        y1 = float(ys.min())
        y2 = float(ys.max() + 1)

        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        side = max(box_w, box_h)
        side *= 1.0 + 2.0 * MASK_LANDMARK_MARGIN_RATIO

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        square_side_int = max(1, int(np.ceil(side)))
        square_x1 = int(round(cx - square_side_int / 2.0))
        square_y1 = int(round(cy - square_side_int / 2.0))
        square_x2 = square_x1 + square_side_int
        square_y2 = square_y1 + square_side_int

        crop_x1 = max(0, square_x1)
        crop_y1 = max(0, square_y1)
        crop_x2 = min(width, square_x2)
        crop_y2 = min(height, square_y2)

        crop = mask_bool[crop_y1:crop_y2, crop_x1:crop_x2].astype(np.float32)
        square = np.zeros((square_side_int, square_side_int), dtype=np.float32)
        paste_x = crop_x1 - square_x1
        paste_y = crop_y1 - square_y1
        paste_x2 = min(square_side_int, paste_x + crop.shape[1])
        paste_y2 = min(square_side_int, paste_y + crop.shape[0])
        src_w = max(0, paste_x2 - paste_x)
        src_h = max(0, paste_y2 - paste_y)
        if src_w > 0 and src_h > 0:
            square[paste_y:paste_y2, paste_x:paste_x2] = crop[:src_h, :src_w]

        processed = cv2.resize(
            square,
            (MASK_LANDMARK_INPUT_SIZE, MASK_LANDMARK_INPUT_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        return processed, (float(square_x1), float(square_y1)), float(square_side_int)


class ProlificSamHelper:
    """Wraps model loading plus CSV-point and automatic mask generation."""

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
        self.mask_landmark_detector: Optional[OnnxMaskLandmarkDetector] = None

    def generate_initial_mask(
        self, image_rgb: np.ndarray, landmarks_px: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if landmarks_px is not None and len(landmarks_px) > 0:
            mask = self.mask_from_landmarks(image_rgb, landmarks_px)
            if mask is not None and mask.any():
                return mask

        return self._generate_from_automatic(image_rgb)

    def mask_from_landmarks(
        self, image_rgb: np.ndarray, landmarks_px: np.ndarray
    ) -> Optional[np.ndarray]:
        return self._mask_from_points(image_rgb, landmarks_px)

    def detect_landmarks_from_mask(self, mask: np.ndarray) -> Optional[np.ndarray]:
        if self.mask_landmark_detector is None:
            self.mask_landmark_detector = OnnxMaskLandmarkDetector()
        return self.mask_landmark_detector.detect(mask)

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

    def _generate_from_automatic(self, image_rgb: np.ndarray) -> np.ndarray:
        original_h, original_w = image_rgb.shape[:2]
        max_side = max(original_h, original_w)
        if max_side > AUTOMATIC_MAX_SIDE:
            scale = AUTOMATIC_MAX_SIDE / float(max_side)
            work_w = max(1, int(round(original_w * scale)))
            work_h = max(1, int(round(original_h * scale)))
            work_image = cv2.resize(image_rgb, (work_w, work_h), interpolation=cv2.INTER_AREA)
        else:
            work_image = image_rgb

        masks = self.automatic_generator.generate(work_image)
        best_mask = self._select_best_mask(masks, work_image.shape[:2])
        if best_mask is None:
            return np.zeros((original_h, original_w), dtype=bool)
        segmentation = best_mask.get("segmentation")
        if segmentation is None:
            return np.zeros((original_h, original_w), dtype=bool)

        mask = segmentation.astype(np.uint8)
        if mask.shape[:2] != (original_h, original_w):
            mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        return mask.astype(bool)

    def _select_best_mask(self, masks: Sequence[dict], shape: Sequence[int]) -> Optional[dict]:
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
        sam_helper: ProlificSamHelper,
        on_mask_updated: Callable[[MaskRecord], None],
        on_landmarks_saved: Callable[[MaskRecord], bool],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.record = record
        self.sam_helper = sam_helper
        self.on_mask_updated = on_mask_updated
        self.on_landmarks_saved = on_landmarks_saved
        self.predictor = self.sam_helper.create_predictor()
        self.predictor.set_image(self.record.image_rgb)
        self.current_mask = record.mask.copy()
        self.current_landmarks = (
            record.landmarks_px.copy() if record.landmarks_px is not None else None
        )
        self.selected_landmark_index: Optional[int] = None
        self.undo_stack: List[np.ndarray] = []
        self.redo_stack: List[np.ndarray] = []

        self.setWindowTitle(self.record.image_path.name)
        self.resize(1000, 600)

        self.original_label = ImageClickLabel()
        self.mask_label = ImageClickLabel()

        info_label = QLabel(
            "SAM: left click adds segments, right click removes.\n"
            "Flood fill: left click fills black, right click fills white.\n"
            "Pen: left click draws white, right click draws black.\n"
            "Landmarks: drag points to reposition them."
        )
        self.mode_selector = QComboBox()
        self.mode_selector.addItems([EDIT_MODE_SAM, EDIT_MODE_FLOOD, EDIT_MODE_PEN, EDIT_MODE_LANDMARKS])
        self.mode_selector.setToolTip("Select how clicks modify the mask.")

        undo_btn = QPushButton("Undo")
        redo_btn = QPushButton("Redo")
        self.undo_button = undo_btn
        self.redo_button = redo_btn
        grow_btn = QPushButton("Grow 1 px")
        shrink_btn = QPushButton("Shrink 1 px")
        detect_landmarks_btn = QPushButton("Detect landmarks from mask")
        regenerate_btn = QPushButton("Regenerate from landmarks")
        save_landmarks_btn = QPushButton("Save Landmarks")
        reset_btn = QPushButton("Reset to auto mask")
        close_btn = QPushButton("Close editor")
        self.save_landmarks_button = save_landmarks_btn

        undo_btn.setToolTip("Undo last mask change (up to 3).")
        redo_btn.setToolTip("Redo last undone change (up to 3).")
        grow_btn.setToolTip("Expand white mask regions by 1 pixel.")
        shrink_btn.setToolTip("Contract white mask regions by 1 pixel.")
        detect_landmarks_btn.setToolTip("Run the mask-to-landmarks ONNX model on the current mask.")
        regenerate_btn.setToolTip("Rebuild the mask using the current landmark positions.")
        save_landmarks_btn.setToolTip("Save adjusted landmarks only when the CSV row was missing them.")
        self.mode_selector.currentTextChanged.connect(lambda _: self._refresh_views())
        reset_btn.clicked.connect(self._handle_reset)
        undo_btn.clicked.connect(self._handle_undo)
        redo_btn.clicked.connect(self._handle_redo)
        grow_btn.clicked.connect(self._handle_grow)
        shrink_btn.clicked.connect(self._handle_shrink)
        detect_landmarks_btn.clicked.connect(self._handle_detect_landmarks_from_mask)
        regenerate_btn.clicked.connect(self._handle_regenerate_from_landmarks)
        save_landmarks_btn.clicked.connect(self._handle_save_landmarks)
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
        button_layout.addWidget(detect_landmarks_btn)
        button_layout.addWidget(regenerate_btn)
        button_layout.addWidget(save_landmarks_btn)
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
        self._update_landmark_buttons()

    def _handle_click(self, x: float, y: float, button: Qt.MouseButton) -> None:
        mode = self.mode_selector.currentText()
        if mode == EDIT_MODE_LANDMARKS:
            self._select_nearest_landmark(x, y)
            self._refresh_views()
            return
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
        if self.mode_selector.currentText() == EDIT_MODE_LANDMARKS:
            self._move_selected_landmark(x, y)
            return
        if self.mode_selector.currentText() != EDIT_MODE_PEN:
            return
        draw_white = button == Qt.LeftButton
        self._apply_new_mask(draw_on_mask(self.current_mask, (x, y), PEN_BRUSH_RADIUS, draw_white))

    def _select_nearest_landmark(self, x: float, y: float) -> None:
        if self.current_landmarks is None or len(self.current_landmarks) == 0:
            self.selected_landmark_index = None
            return
        deltas = self.current_landmarks - np.array([x, y], dtype=np.float32)
        distances = np.sqrt((deltas * deltas).sum(axis=1))
        nearest = int(np.argmin(distances))
        if float(distances[nearest]) <= LANDMARK_HIT_RADIUS:
            self.selected_landmark_index = nearest

    def _move_selected_landmark(self, x: float, y: float) -> None:
        if self.current_landmarks is None:
            return
        if self.selected_landmark_index is None:
            self._select_nearest_landmark(x, y)
            if self.selected_landmark_index is None:
                return
        height, width = self.record.image_rgb.shape[:2]
        updated = self.current_landmarks.copy()
        updated[self.selected_landmark_index, 0] = np.clip(x, 0, max(width - 1, 0))
        updated[self.selected_landmark_index, 1] = np.clip(y, 0, max(height - 1, 0))
        self.current_landmarks = updated
        self.record.update_landmarks(updated)
        self._refresh_views()
        self._update_landmark_buttons()

    def _handle_regenerate_from_landmarks(self) -> None:
        if self.current_landmarks is None or len(self.current_landmarks) == 0:
            QMessageBox.information(self, "Landmarks", "No landmarks are available for this image.")
            return
        mask = self.sam_helper.mask_from_landmarks(self.record.image_rgb, self.current_landmarks)
        if mask is None or not mask.any():
            QMessageBox.warning(self, "Landmarks", "Could not generate a mask from these landmarks.")
            return
        self._apply_new_mask(mask.astype(bool))

    def _handle_detect_landmarks_from_mask(self) -> None:
        if not self.current_mask.any():
            QMessageBox.information(self, "Landmarks", "The current mask is empty.")
            return

        reply = QMessageBox.question(
            self,
            "Detect landmarks",
            "Run mask-to-landmarks detection using the current mask?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            landmarks = self.sam_helper.detect_landmarks_from_mask(self.current_mask)
        except Exception as exc:
            QMessageBox.warning(self, "Landmarks", f"Mask landmark detection failed:\n{exc}")
            return

        if landmarks is None or len(landmarks) == 0:
            landmarks = default_hand_landmarks(self.record.image_rgb.shape, self.record.image_path.name)
            QMessageBox.information(
                self,
                "Landmarks",
                "The mask model did not return landmarks, so a movable default template was placed instead.",
            )

        self.current_landmarks = landmarks.astype(np.float32)
        self.record.landmarks_source = "mask_onnx"
        self.record.update_landmarks(self.current_landmarks)
        self.selected_landmark_index = None
        self.mode_selector.setCurrentText(EDIT_MODE_LANDMARKS)
        self._refresh_views()
        self._update_landmark_buttons()

    def _handle_save_landmarks(self) -> None:
        if self.current_landmarks is None or len(self.current_landmarks) == 0:
            QMessageBox.information(self, "Landmarks", "No landmarks are available to save.")
            return
        if not self.record.landmarks_originally_missing:
            QMessageBox.information(
                self,
                "Landmarks",
                "This image already had CSV landmarks, so they were not overwritten.",
            )
            return
        changed = (
            self.record.landmarks_px is None
            or not np.allclose(self.record.landmarks_px, self.current_landmarks)
        )
        if changed:
            self.record.update_landmarks(self.current_landmarks)
        if self.on_landmarks_saved(self.record):
            self.record.landmarks_originally_missing = False
            self.record.landmarks_dirty = False
            self._update_landmark_buttons()
            QMessageBox.information(self, "Landmarks", "Saved landmarks to the CSV.")
        else:
            QMessageBox.warning(self, "Landmarks", "Failed to save landmarks to the CSV.")

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

    def _update_landmark_buttons(self) -> None:
        has_landmarks = self.current_landmarks is not None and len(self.current_landmarks) > 0
        self.save_landmarks_button.setEnabled(
            bool(has_landmarks and self.record.landmarks_originally_missing)
        )

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
        if self.mode_selector.currentText() == EDIT_MODE_LANDMARKS and self.current_landmarks is not None:
            overlay = draw_landmarks_overlay(
                overlay,
                self.current_landmarks,
                selected_index=self.selected_landmark_index,
            )
        self.original_label.set_pixmap(numpy_to_qpixmap(overlay))
        self.mask_label.set_pixmap(numpy_to_qpixmap(mask_to_rgb(self.current_mask)))


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


def draw_landmarks_overlay(
    image_rgb: np.ndarray,
    landmarks_px: np.ndarray,
    *,
    selected_index: Optional[int] = None,
) -> np.ndarray:
    output = image_rgb.copy()
    points = np.asarray(landmarks_px, dtype=np.float32)
    for idx, point in enumerate(points):
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        color = (255, 220, 0) if idx != selected_index else (255, 80, 80)
        cv2.circle(output, (x, y), LANDMARK_RADIUS + 2, (0, 0, 0), thickness=-1)
        cv2.circle(output, (x, y), LANDMARK_RADIUS, color, thickness=-1)
        cv2.circle(output, (x, y), LANDMARK_RADIUS + 2, (255, 255, 255), thickness=2)
    return output


def default_hand_landmarks(image_shape: Sequence[int], image_name: str) -> np.ndarray:
    image_h, image_w = image_shape[:2]
    center = np.array([image_w * 0.5, image_h * 0.52], dtype=np.float32)
    scale = min(image_w, image_h) * DEFAULT_HAND_TEMPLATE_SCALE

    # Canonical right-hand dorsal-ish pose in relative x/y coordinates.
    points = np.array(
        [
            [0.00, 0.58],   # wrist
            [-0.22, 0.30],
            [-0.38, 0.08],
            [-0.52, -0.10],
            [-0.66, -0.25],
            [-0.20, 0.00],
            [-0.25, -0.35],
            [-0.27, -0.60],
            [-0.29, -0.82],
            [0.00, -0.04],
            [0.00, -0.43],
            [0.00, -0.72],
            [0.00, -0.96],
            [0.20, 0.00],
            [0.25, -0.36],
            [0.28, -0.62],
            [0.30, -0.84],
            [0.38, 0.10],
            [0.50, -0.18],
            [0.58, -0.38],
            [0.66, -0.56],
        ],
        dtype=np.float32,
    )

    lower_name = image_name.lower()
    if lower_name.startswith("left") or "_left" in lower_name:
        points[:, 0] *= -1.0

    points_px = points * scale + center
    points_px[:, 0] = np.clip(points_px[:, 0], 0, max(image_w - 1, 0))
    points_px[:, 1] = np.clip(points_px[:, 1], 0, max(image_h - 1, 0))
    return points_px.astype(np.float32)


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


def load_existing_mask(mask_path: Path) -> Optional[np.ndarray]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        warnings.warn(f"Failed to read cached mask: {mask_path}", RuntimeWarning)
        return None
    return mask > 127


def load_image_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is not None:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    try:
        import pillow_heif
        from PIL import Image, ImageOps

        pillow_heif.register_heif_opener()
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            return np.asarray(image)
    except Exception as exc:
        raise FileNotFoundError(f"Failed to read image: {path}") from exc


def save_mask_to_disk(mask: np.ndarray, output_path: Path) -> None:
    mask_u8 = np.where(mask, 255, 0).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), mask_u8):
        raise RuntimeError("cv2.imwrite returned False")


@dataclass(frozen=True)
class ProlificImageEntry:
    image_path: Path
    prolific_pid: str
    landmarks_px: Optional[np.ndarray]
    landmarks_px_xyz: Optional[np.ndarray]
    has_detection: bool
    row_index: Optional[int]
    landmarks_originally_missing: bool


class ProlificThumbnailGridWindow(QWidget):
    """Main window showing one participant folder and saving Prolific masks."""

    def __init__(
        self,
        records: Sequence[MaskRecord],
        sam_helper: ProlificSamHelper,
        landmark_store: ProlificLandmarkStore,
        participant_folder: Path,
        mask_subdir: str,
    ) -> None:
        super().__init__()
        self.records = list(records)
        self.sam_helper = sam_helper
        self.landmark_store = landmark_store
        self.participant_folder = participant_folder
        self.mask_subdir = mask_subdir
        self.editors: List[MaskEditorWindow] = []
        self.setWindowTitle(f"Prolific Mask Review - {participant_folder.name}")
        self.resize(1400, 900)

        header_label = QLabel(participant_folder.name)
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
        editor = MaskEditorWindow(
            record,
            self.sam_helper,
            self._handle_mask_update,
            self._handle_landmarks_save,
        )
        editor.setAttribute(Qt.WA_DeleteOnClose)
        editor.show()
        self.editors.append(editor)

    def _handle_mask_update(self, record: MaskRecord) -> None:
        index = self.records.index(record)
        pixmap = build_thumbnail_pixmap(record, THUMBNAIL_SIZE)
        button = self.thumbnail_buttons[index]
        button.setIcon(QIcon(pixmap))
        button.setIconSize(pixmap.size())

    def _handle_landmarks_save(self, record: MaskRecord) -> bool:
        return self.landmark_store.save_missing_landmarks(record)

    def _save_all_masks(self) -> None:  # pragma: no cover - GUI runtime
        failures: List[str] = []
        saved_paths: List[Path] = []

        for record in self.records:
            output_path = build_output_path(record.image_path, self.mask_subdir)
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
            return

        QMessageBox.information(
            self,
            "Save masks",
            f"Saved {len(saved_paths)} masks.\nExample: {saved_paths[0]}"
            if saved_paths
            else "No masks saved.",
        )


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


def parse_landmarks_px_xyz(raw_value: str) -> Optional[np.ndarray]:
    parsed = parse_json_like(raw_value)
    if parsed is None:
        return None

    points = np.asarray(parsed, dtype=np.float32)
    if points.size == 0 or points.ndim != 2 or points.shape[1] < 3:
        return None

    points = points[:, :3]
    points = points[np.isfinite(points).all(axis=1)]
    if points.size == 0:
        return None
    return points


def row_has_detection(row: dict) -> bool:
    value = str(row.get("has_detection", "")).strip().lower()
    return value in {"1", "true", "yes", "y"}


def resolve_csv_image_path(row: dict, dataset_root: Path) -> Optional[Path]:
    candidates: List[Path] = []
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
            return candidate.resolve()
    return None


def image_relative_path(image_path: Path, dataset_root: Path) -> str:
    participants_dir = dataset_root / "participants"
    try:
        return str(image_path.resolve().relative_to(participants_dir.resolve()))
    except ValueError:
        try:
            return str(image_path.resolve().relative_to(dataset_root.resolve()))
        except ValueError:
            return image_path.name


def format_landmarks_px(points_xy: np.ndarray) -> str:
    rounded = np.round(np.asarray(points_xy, dtype=np.float32), 5).tolist()
    return json.dumps(rounded, separators=(",", ":"))


def format_landmarks_px_xyz(points_xyz: Optional[np.ndarray], points_xy: np.ndarray) -> str:
    if points_xyz is None or points_xyz.shape[0] != points_xy.shape[0]:
        z = np.zeros((points_xy.shape[0], 1), dtype=np.float32)
        points_xyz = np.concatenate([points_xy.astype(np.float32), z], axis=1)
    else:
        points_xyz = points_xyz.astype(np.float32).copy()
        points_xyz[:, :2] = points_xy
    rounded = np.round(points_xyz, 5).tolist()
    return json.dumps(rounded, separators=(",", ":"))


class ProlificLandmarkStore:
    def __init__(self, dataset_root: Path, csv_path: Path, include_failed_detections: bool) -> None:
        self.dataset_root = dataset_root
        self.csv_path = csv_path
        self.include_failed_detections = include_failed_detections
        self.fieldnames: List[str] = []
        self.rows: List[dict] = []
        self.entries: Dict[Path, ProlificImageEntry] = {}

    def load(self) -> Dict[Path, ProlificImageEntry]:
        self.entries = {}
        self.rows = []

        if not self.csv_path.exists():
            warnings.warn(f"Landmark CSV not found: {self.csv_path}", RuntimeWarning)
            return self.entries

        with self.csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None:
                return self.entries
            self.fieldnames = list(reader.fieldnames)
            self.rows = list(reader)

        for row_index, row in enumerate(self.rows):
            image_path = resolve_csv_image_path(row, self.dataset_root)
            if image_path is None:
                continue

            has_detection = row_has_detection(row)
            landmarks_px = parse_landmarks_px(str(row.get("rgb_landmarks_px", "")))
            landmarks_px_xyz = parse_landmarks_px_xyz(str(row.get("rgb_landmarks_px_xyz", "")))
            if not has_detection and not self.include_failed_detections:
                usable_landmarks = None
            else:
                usable_landmarks = landmarks_px

            self.entries[image_path] = ProlificImageEntry(
                image_path=image_path,
                prolific_pid=str(row.get("prolific_pid", "")).strip(),
                landmarks_px=usable_landmarks,
                landmarks_px_xyz=landmarks_px_xyz,
                has_detection=has_detection,
                row_index=row_index,
                landmarks_originally_missing=landmarks_px is None,
            )

        return self.entries

    def save_missing_landmarks(self, record: MaskRecord) -> bool:
        if record.landmarks_px is None or not record.landmarks_originally_missing:
            return False

        self._ensure_fieldnames()
        resolved_path = record.image_path.resolve()
        entry = self.entries.get(resolved_path)
        row: dict
        if entry is not None and entry.row_index is not None:
            row = self.rows[entry.row_index]
        else:
            row = self._new_row(record.image_path)
            self.rows.append(row)
            entry = ProlificImageEntry(
                image_path=resolved_path,
                prolific_pid=record.image_path.parent.name,
                landmarks_px=None,
                landmarks_px_xyz=None,
                has_detection=False,
                row_index=len(self.rows) - 1,
                landmarks_originally_missing=True,
            )

        row["rgb_landmarks_px"] = format_landmarks_px(record.landmarks_px)
        row["rgb_landmarks_px_xyz"] = format_landmarks_px_xyz(
            record.landmarks_px_xyz,
            record.landmarks_px,
        )
        row["has_detection"] = "true"
        row["status"] = "manual_landmarks" if record.landmarks_dirty else "fallback_detected"
        row["error"] = ""
        row["processed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")

        image_h, image_w = record.image_rgb.shape[:2]
        row["image_width"] = str(image_w)
        row["image_height"] = str(image_h)
        bbox = landmarks_to_bbox(record.landmarks_px, image_w, image_h)
        row["bbox_px"] = json.dumps(bbox, separators=(",", ":"))

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)

        self.entries[resolved_path] = ProlificImageEntry(
            image_path=resolved_path,
            prolific_pid=record.image_path.parent.name,
            landmarks_px=record.landmarks_px.copy(),
            landmarks_px_xyz=record.landmarks_px_xyz.copy() if record.landmarks_px_xyz is not None else None,
            has_detection=True,
            row_index=entry.row_index,
            landmarks_originally_missing=False,
        )
        return True

    def _ensure_fieldnames(self) -> None:
        if not self.fieldnames:
            self.fieldnames = [
                "prolific_pid",
                "relative_path",
                "image_path",
                "filename",
                "hand",
                "orientation",
                "image_index",
                "image_width",
                "image_height",
                "rgb_landmarks_px",
                "rgb_landmarks_px_xyz",
                "bbox_px",
                "has_detection",
                "status",
                "error",
                "processed_at",
            ]
            return

        for name in (
            "rgb_landmarks_px",
            "rgb_landmarks_px_xyz",
            "bbox_px",
            "has_detection",
            "status",
            "error",
            "processed_at",
        ):
            if name not in self.fieldnames:
                self.fieldnames.append(name)

    def _new_row(self, image_path: Path) -> dict:
        row = {name: "" for name in self.fieldnames}
        row["prolific_pid"] = image_path.parent.name
        row["relative_path"] = image_relative_path(image_path, self.dataset_root)
        row["image_path"] = str(image_path)
        row["filename"] = image_path.name
        stem_parts = image_path.stem.split("_")
        if len(stem_parts) >= 3:
            row["hand"] = stem_parts[0]
            row["orientation"] = stem_parts[1]
            row["image_index"] = stem_parts[2]
        return row


def landmarks_to_bbox(points_xy: np.ndarray, image_w: int, image_h: int) -> list[int]:
    points = np.asarray(points_xy, dtype=np.float32)
    x1 = int(np.clip(np.floor(points[:, 0].min()), 0, max(image_w - 1, 0)))
    y1 = int(np.clip(np.floor(points[:, 1].min()), 0, max(image_h - 1, 0)))
    x2 = int(np.clip(np.ceil(points[:, 0].max()), 0, max(image_w - 1, 0)))
    y2 = int(np.clip(np.ceil(points[:, 1].max()), 0, max(image_h - 1, 0)))
    return [x1, y1, x2, y2]


def load_prolific_entries(
    dataset_root: Path,
    csv_path: Path,
    *,
    include_failed_detections: bool = False,
) -> Dict[Path, ProlificImageEntry]:
    return ProlificLandmarkStore(
        dataset_root,
        csv_path,
        include_failed_detections=include_failed_detections,
    ).load()


def iter_participant_image_files(participant_folder: Path) -> Iterable[Path]:
    for path in sorted(participant_folder.iterdir(), key=lambda p: p.name.lower()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def build_output_path(image_path: Path, mask_subdir: str = MASK_SUBDIR) -> Path:
    output_name = f"{MASK_PREFIX}{image_path.stem}{MASK_EXTENSION}"
    if mask_subdir:
        return image_path.parent / mask_subdir / output_name
    return image_path.with_name(output_name)


def list_participant_folders(root: Path) -> List[Path]:
    participants_dir = root / "participants"
    if not participants_dir.is_dir():
        return []
    return sorted(
        [p for p in participants_dir.iterdir() if p.is_dir()],
        key=lambda p: p.name.lower(),
    )


def list_folders_missing_masks(root: Path, mask_subdir: str) -> List[Path]:
    missing: List[Path] = []
    for folder in list_participant_folders(root):
        images = list(iter_participant_image_files(folder))
        if not images:
            continue
        if any(not build_output_path(image_path, mask_subdir).exists() for image_path in images):
            missing.append(folder)
    return missing


def summarize_folder_coverage(root: Path, mask_subdir: str) -> tuple[int, int, int]:
    total = 0
    with_masks = 0
    without_masks = 0
    for folder in list_participant_folders(root):
        images = list(iter_participant_image_files(folder))
        if not images:
            continue
        total += 1
        needs_masks = any(
            not build_output_path(image_path, mask_subdir).exists() for image_path in images
        )
        if needs_masks:
            without_masks += 1
        else:
            with_masks += 1
    return total, with_masks, without_masks


def prompt_for_participant_folder(
    root: Path,
    missing_folders: Sequence[Path],
    limit: int = LIST_DISPLAY_LIMIT,
) -> Path:
    display_count = min(limit, len(missing_folders))

    if missing_folders:
        print(f"\nParticipants missing masks (showing up to {display_count}):")
        for idx, folder in enumerate(missing_folders[:display_count], start=1):
            print(f"  {idx}. {folder.name}")
        print(
            "\nEnter a number from the list above to open it, "
            "or type a participant id/path."
        )
        print("Press Enter with no input to open the first participant in the list.")
    else:
        print(
            "\nAll participant folders appear to contain masks. "
            "Type a participant id/path to open one."
        )

    participants_dir = root / "participants"
    while True:
        choice = input("Participant selection: ").strip()
        if not choice:
            if missing_folders:
                return missing_folders[0]
            print("Please enter a participant id or absolute path.")
            continue

        if choice.isdigit() and display_count > 0:
            idx = int(choice)
            if 1 <= idx <= display_count:
                return missing_folders[idx - 1]
            print(f"Enter a number between 1 and {display_count}.")
            continue

        candidate = Path(choice).expanduser()
        if not candidate.is_absolute():
            candidate = participants_dir / choice
        if candidate.is_dir():
            return candidate
        print(f"Participant folder not found: {candidate}. Try again.")


def build_records(
    helper: ProlificSamHelper,
    image_paths: Sequence[Path],
    entries: Dict[Path, ProlificImageEntry],
    mask_subdir: str,
) -> List[MaskRecord]:
    records: List[MaskRecord] = []
    total = len(image_paths)

    for idx, image_path in enumerate(image_paths, start=1):
        image_rgb = load_image_rgb(image_path)
        resolved_path = image_path.resolve()
        entry = entries.get(resolved_path)
        landmarks_px = entry.landmarks_px.copy() if entry and entry.landmarks_px is not None else None
        landmarks_px_xyz = (
            entry.landmarks_px_xyz.copy()
            if entry and entry.landmarks_px_xyz is not None
            else None
        )
        landmarks_originally_missing = bool(entry is None or entry.landmarks_originally_missing)
        landmarks_source = "csv" if landmarks_px is not None else ""
        mask: Optional[np.ndarray] = None
        from_disk = False

        cached_mask_path = build_output_path(image_path, mask_subdir)
        if cached_mask_path.exists():
            mask = load_existing_mask(cached_mask_path)
            if mask is not None:
                from_disk = True
                print(f"[{idx}/{total}] Loaded cached mask {cached_mask_path.name}")

        if mask is None:
            landmark_note = {
                "csv": "CSV landmarks",
                "fallback_detected": "fallback detection",
                "default_template": "default landmark template",
            }.get(landmarks_source, "fallback detection")
            print(f"[{idx}/{total}] Generating mask for {image_path.name} ({landmark_note})")
            mask = helper.generate_initial_mask(
                image_rgb,
                landmarks_px,
            )

        records.append(
            MaskRecord(
                image_path=image_path,
                image_rgb=image_rgb,
                mask=mask.copy(),
                auto_mask=mask.copy(),
                landmarks_px=landmarks_px.copy() if landmarks_px is not None else None,
                landmarks_px_xyz=landmarks_px_xyz.copy() if landmarks_px_xyz is not None else None,
                landmarks_originally_missing=landmarks_originally_missing,
                landmarks_source=landmarks_source,
                from_disk=from_disk,
            )
        )

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive MobileSAM reviewer for Prolific hand images.")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Prolific dataset root containing participants/ and prolific_image_landmarks.csv.",
    )
    parser.add_argument(
        "--participant",
        "--folder",
        dest="participant",
        type=str,
        help="Participant id or folder path to open. Defaults to prompting.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Landmark CSV path. Defaults to <root>/prolific_image_landmarks.csv.",
    )
    parser.add_argument(
        "--mask-subdir",
        type=str,
        default=MASK_SUBDIR,
        help="Subfolder inside each participant folder for masks. Use an empty string to save beside images.",
    )
    parser.add_argument(
        "--include-failed-detections",
        action="store_true",
        help="Use CSV landmark rows even when has_detection is not true.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = args.root.expanduser()
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}")

    csv_path = args.csv.expanduser() if args.csv is not None else root / DEFAULT_CSV_NAME
    if not csv_path.is_absolute():
        csv_path = root / csv_path

    mask_subdir = args.mask_subdir.strip()

    if args.participant:
        candidate = Path(args.participant).expanduser()
        if not candidate.is_absolute():
            candidate = root / "participants" / args.participant
        if not candidate.is_dir():
            raise SystemExit(f"Participant folder not found: {candidate}")
        target_folder = candidate
    else:
        missing = list_folders_missing_masks(root, mask_subdir)
        total, with_masks, without_masks = summarize_folder_coverage(root, mask_subdir)
        if total > 0:
            pct_with = (with_masks / total) * 100.0
            pct_without = (without_masks / total) * 100.0
            print(
                f"\nParticipant coverage summary: {with_masks}/{total} ({pct_with:.1f}%) with masks, "
                f"{without_masks}/{total} ({pct_without:.1f}%) without masks."
            )
        target_folder = prompt_for_participant_folder(root, missing, LIST_DISPLAY_LIMIT)

    image_paths = list(iter_participant_image_files(target_folder))
    if not image_paths:
        raise SystemExit(f"No image files found in participant folder: {target_folder}")

    print(f"\nLoading landmark index from: {csv_path}")
    landmark_store = ProlificLandmarkStore(
        root,
        csv_path,
        include_failed_detections=args.include_failed_detections,
    )
    entries = landmark_store.load()
    print(f"Loaded landmark rows for {len(entries)} images.")

    print(f"\nOpening participant folder: {target_folder}")
    print(f"Found {len(image_paths)} images.")
    helper = ProlificSamHelper()
    records = build_records(helper, image_paths, entries, mask_subdir)

    app = QApplication(sys.argv)
    window = ProlificThumbnailGridWindow(records, helper, landmark_store, target_folder, mask_subdir)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
