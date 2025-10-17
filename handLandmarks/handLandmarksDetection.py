

# === Standard Library ===
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, List
import numpy as np

# === Third-party Libraries ===
import cv2
import onnxruntime as ort
import mediapipe as mp
from huggingface_hub import hf_hub_download

# === MediaPipe Tasks ===
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker
from mediapipe.tasks.python.core import base_options as mp_base


class SentisHandLandmarkDetector:
    """
    Detector for Unity-Sentis hand landmarks.

    >>> det = SentisHandLandmarkDetector()
    >>> lms_px = det.detect(img)                # ← unchanged
    >>> lms_n  = det.normalize(lms_px)          # ← NEW helper
    >>>             #  x, y now ∈ [0 … 1] w.r.t. the full image
    """

    def __init__(self):
        self.lmk_sess = ort.InferenceSession(
            hf_hub_download("unity/sentis-hand-landmark", "hand_landmark.onnx"),
            providers=["CPUExecutionProvider"],
        )

    # ------------------------------------------------------------------ #
    # original API – returns pixel values in the 224×224 inference frame
    # ------------------------------------------------------------------ #
    def detect(self, img):
        """
        Returns
        -------
        landmarks_px : (21, 3) ndarray
                       x, y ,z normalized
        """
        lmk_in = (
            np.transpose(cv2.resize(img, (224, 224)), (2, 0, 1))[None] / 255.0
        ).astype(np.float32)
        outs = self.lmk_sess.run(None, {self.lmk_sess.get_inputs()[0].name: lmk_in})

        lms = next(
            (
                np.asarray(a).ravel().reshape(21, 3)
                for a in outs
                if np.asarray(a).size == 63
            ),
            None,
        )
        if lms is None:
            raise ValueError("No landmark output of size 63")
        lms_norm = SentisHandLandmarkDetector.normalize(lms)
        return lms_norm,lms

    # ------------------------------------------------------------------ #
    # NEW helper – normalise x,y to the full image (0 … 1)
    # ------------------------------------------------------------------ #
    @staticmethod
    def normalize(landmarks_px: np.ndarray) -> np.ndarray:
        """
        Convert the detector’s pixel-space output to **image-normalised
        coordinates** (x, y ∈ [0 .. 1]) so it plugs straight into your new
        DisplayUtils.

        Parameters
        ----------
        landmarks_px : (21, 3) ndarray
            Raw output of `detect`.

        Returns
        -------
        landmarks_norm : (21, 3) float32
            landmarks_norm[:,0:2]  ∈ [0, 1]  (full-image normalised)
            landmarks_norm[:,  2]  unchanged (z in model units)
        """
        if landmarks_px.shape != (21, 3):
            raise ValueError("landmarks must have shape (21, 3)")

        lms = landmarks_px.astype(np.float32).copy()
        lms[:, 0:2] /= 224.0          # 224×224 is the model’s input size
        return lms

class MediaPipeTaskHandLandmarkDetector:
    """
    detect(img)  →  (landmarks_norm_img,  bbox_px)

        landmarks_norm_img : (21 × 3) float32
                              x, y ∈ [0 .. 1]  **w.r.t. full image**
                              z   keeps MediaPipe’s “image-width” units
        bbox_px            : (xmin, ymin, xmax, ymax)  ints (pixels)
    """

    _MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

    # ------------------------------------------------------------------ #
    # initialise the MediaPipe Hand-Landmarker task
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        num_hands: int = 1,
        det_conf: float = 0.5,
        pres_conf: float = 0.5,
        track_conf: float = 0.5,
    ) -> None:

        if not self._MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {self._MODEL_PATH}\n"
                "Download hand_landmarker.task and place it next to this .py file."
            )

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self._MODEL_PATH)),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=det_conf,
            min_hand_presence_confidence=pres_conf,
            min_tracking_confidence=track_conf,
        )
        self._lm = HandLandmarker.create_from_options(options)

    # ------------------------------------------------------------------ #
    # helper – convert MP landmarks to image-normalised space
    # ------------------------------------------------------------------ #
    @staticmethod
    def to_array(
        lm_list: List[mp.tasks.components.containers.Landmark],
    ) -> np.ndarray:
        """
        Convert MediaPipe landmarks to an (N×3) float32 array:
        - x,y in [0,1] relative to the full image
        - z unchanged (image-width units)
        """
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in lm_list],
            dtype=np.float32
        )

    
    # ------------------------------------------------------------------ #
    # helper – convert MP landmarks to image space
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_image_space(
        lm_list: List[mp.tasks.components.containers.Landmark],
        image_width: int,
        image_height: int,
    ) -> np.ndarray:
        """
        Return (N × 3) float32 array with:
          - x,y in pixel coordinates (x in [0,image_width], y in [0,image_height])
          - z in MediaPipe’s original scale (image-width units).
        """
        pts = []
        for lm in lm_list:
            # scale normalized coords to image pixels
            x_px = lm.x * image_width
            y_px = lm.y * image_height
            z_px = lm.z * image_width
            pts.append((x_px, y_px, z_px))

        return np.asarray(pts, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # API – detect one frame
    # ------------------------------------------------------------------ #
    def detect(
        self, img_bgr: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Parameters
        ----------
        img_bgr : np.ndarray (BGR image)

        Returns
        -------
        landmarks_norm_img : (21 × 3) float32, x,y ∈ [0,1] in full image
        bbox_px            : (xmin, ymin, xmax, ymax) pixel ints
        """
        img_h, img_w = img_bgr.shape[:2]

        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        )
        res = self._lm.detect(mp_img)

        if not res.hand_landmarks:
            return np.empty((0, 3), np.float32), (0, 0, 0, 0)

        lm_abs = res.hand_landmarks[0]  # first hand

        # ---- landmarks normalised to entire image --------------------- #
        lms_norm = self.to_array(lm_abs)
        lms_image = self._to_image_space(lm_abs,img_w,img_h)

        return lms_norm,lms_image

