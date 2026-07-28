"""Detector backend implementations for Phase 04B YOLO26 Semantic Fusion."""

from __future__ import annotations

import hashlib
import time
from typing import Sequence, Tuple

import cv2
import numpy as np

from detector_interfaces import Detection, DetectionResult, ObjectDetector
from semantic_fusion import RETAINED_CLASSES


class NoneObjectDetector(ObjectDetector):
    """Empty detector backend reproducing original non-semantic behavior."""

    def infer(self, left_bgr: np.ndarray) -> DetectionResult:
        h, w = left_bgr.shape[:2]
        return DetectionResult(
            detections=(),
            backend="none",
            precision="none",
            input_shape=(1, 3, h, w),
            model_sha256="",
            preprocess_ms=0.0,
            inference_ms=0.0,
            postprocess_ms=0.0,
        )

    def close(self) -> None:
        pass


def _ensure_cuda_dlls_on_path() -> None:
    """Prepend torch's bundled CUDA/cuDNN DLLs to PATH so ORT's CUDA EP loads.

    PyTorch's CUDA wheels ship cuDNN 9.* and CUDA 12.* DLLs inside
    ``torch/lib`` but do not put that directory on ``PATH``. ONNX Runtime's
    CUDA execution provider looks them up via ``LoadLibrary`` and, when they
    are missing, fails with ``LoadLibrary error 126`` and silently falls back
    to CPU. Adding torch's lib dir to ``PATH`` before creating the session
    makes the CUDA provider load successfully.
    """
    import os

    try:
        import torch

        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(torch_lib):
            os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        # torch is an optional dependency for the ONNX-only path; if it is not
        # importable the caller must have set up the CUDA DLLs themselves.
        pass


def letterbox_image(
    img: np.ndarray,
    target_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resize and letterbox image preserving aspect ratio."""
    h, w = img.shape[:2]
    target_h, target_w = target_shape
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    dw = (target_w - new_w) / 2.0
    dh = (target_h - new_h) / 2.0

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, scale, (dw, dh)


class PyTorchYolo26Detector(ObjectDetector):
    """PyTorch / Ultralytics YOLO26 object detector."""

    def __init__(
        self,
        model_path: str = "yolo26n.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_shape: Tuple[int, int] = (640, 640),
        device: str = "cuda:0",
    ) -> None:
        import torch
        from ultralytics import YOLO

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.input_shape = input_shape
        self.device = device if torch.cuda.is_available() else "cpu"

        self.model = YOLO(model_path)
        self.model.to(self.device)

        # Compute model SHA256 if file exists
        self.model_sha256 = ""
        try:
            with open(model_path, "rb") as f:
                self.model_sha256 = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            self.model_sha256 = "builtin_or_remote"

    def infer(self, left_bgr: np.ndarray) -> DetectionResult:
        t0 = time.perf_counter()
        h_orig, w_orig = left_bgr.shape[:2]

        t1 = time.perf_counter()
        results = self.model.predict(
            source=left_bgr,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.input_shape[0],
            device=self.device,
            verbose=False,
        )
        t2 = time.perf_counter()

        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)

                for box, conf, cls_id in zip(xyxy, confs, cls_ids):
                    if cls_id in RETAINED_CLASSES:
                        x0, y0, x1, y1 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                        detections.append(
                            Detection(
                                bbox_xyxy=(x0, y0, x1, y1),
                                class_id=int(cls_id),
                                class_name=RETAINED_CLASSES[int(cls_id)],
                                confidence=float(conf),
                            )
                        )

        t3 = time.perf_counter()

        preprocess_ms = (t1 - t0) * 1000.0
        inference_ms = (t2 - t1) * 1000.0
        postprocess_ms = (t3 - t2) * 1000.0

        return DetectionResult(
            detections=tuple(detections),
            backend="yolo26-pytorch",
            precision="fp32" if "cuda" in self.device else "cpu",
            input_shape=(1, 3, self.input_shape[0], self.input_shape[1]),
            model_sha256=self.model_sha256,
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            postprocess_ms=postprocess_ms,
        )

    def close(self) -> None:
        pass


class ONNXYolo26Detector(ObjectDetector):
    """ONNX Runtime GPU object detector for YOLO26."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_shape: Tuple[int, int] = (640, 640),
        providers: Sequence[str] = ("CUDAExecutionProvider", "CPUExecutionProvider"),
    ) -> None:
        import onnxruntime as ort

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.input_shape = input_shape

        # ONNX Runtime's CUDA EP needs cuDNN 9.* and CUDA 12.* DLLs on PATH.
        # On Windows these are bundled inside torch's lib/ but not added to PATH
        # by default, so ORT silently falls back to CPU. Make them discoverable
        # before creating the session.
        if "CUDAExecutionProvider" in providers:
            _ensure_cuda_dlls_on_path()

        self.session = ort.InferenceSession(model_path, providers=list(providers))
        self.providers = self.session.get_providers()
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        with open(model_path, "rb") as f:
            self.model_sha256 = hashlib.sha256(f.read()).hexdigest()

    def infer(self, left_bgr: np.ndarray) -> DetectionResult:
        t0 = time.perf_counter()
        h_orig, w_orig = left_bgr.shape[:2]

        padded_img, scale, (pad_w, pad_h) = letterbox_image(left_bgr, self.input_shape)
        rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, :]

        t1 = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        t2 = time.perf_counter()

        # YOLO26 end-to-end ONNX output contract (verified on yolo26n.onnx):
        #   shape  : (1, 300, 6)
        #   columns: [x1, y1, x2, y2, score, class_id]
        #   coords : pixel coords in the letterboxed 640x640 space (xyxy, not cxcywh)
        # The old code incorrectly parsed these as [cx, cy, w, h, score, class_id].
        detections = []
        out = outputs[0]  # (1, 300, 6)
        predictions = out[0]  # (300, 6)

        boxes_xyxy = []
        scores_list = []
        cls_list = []

        for pred in predictions:
            x1, y1, x2, y2, conf, cls_id = pred
            if conf < self.confidence_threshold:
                continue
            cls_id_int = int(round(float(cls_id)))
            if cls_id_int not in RETAINED_CLASSES:
                continue
            boxes_xyxy.append([float(x1), float(y1), float(x2), float(y2)])
            scores_list.append(float(conf))
            cls_list.append(cls_id_int)

        if boxes_xyxy:
            boxes_arr = np.array(boxes_xyxy, dtype=np.float32)
            scores_arr = np.array(scores_list, dtype=np.float32)

            # NMS expects [x, y, w, h] format; convert from xyxy.
            boxes_xywh = boxes_arr.copy()
            boxes_xywh[:, 2] = boxes_arr[:, 2] - boxes_arr[:, 0]
            boxes_xywh[:, 3] = boxes_arr[:, 3] - boxes_arr[:, 1]

            indices = cv2.dnn.NMSBoxes(
                boxes_xywh.tolist(),
                scores_arr.tolist(),
                self.confidence_threshold,
                self.iou_threshold,
            )

            if len(indices) > 0:
                for idx in indices.flatten():
                    c_id = cls_list[idx]
                    x1, y1, x2, y2 = boxes_arr[idx]
                    # Map back from letterboxed 640x640 coords to original image pixels.
                    x1 = float(np.clip((x1 - pad_w) / scale, 0.0, w_orig))
                    y1 = float(np.clip((y1 - pad_h) / scale, 0.0, h_orig))
                    x2 = float(np.clip((x2 - pad_w) / scale, 0.0, w_orig))
                    y2 = float(np.clip((y2 - pad_h) / scale, 0.0, h_orig))
                    detections.append(
                        Detection(
                            bbox_xyxy=(x1, y1, x2, y2),
                            class_id=c_id,
                            class_name=RETAINED_CLASSES[c_id],
                            confidence=float(scores_arr[idx]),
                        )
                    )

        t3 = time.perf_counter()

        return DetectionResult(
            detections=tuple(detections),
            backend="yolo26-onnx",
            precision="fp32",
            input_shape=(1, 3, self.input_shape[0], self.input_shape[1]),
            model_sha256=self.model_sha256,
            preprocess_ms=(t1 - t0) * 1000.0,
            inference_ms=(t2 - t1) * 1000.0,
            postprocess_ms=(t3 - t2) * 1000.0,
        )

    def close(self) -> None:
        pass


def get_detector_backend(
    backend_name: str,
    model_path: str | None = None,
    confidence_threshold: float = 0.25,
    precision: str = "fp32",
) -> ObjectDetector:
    """Factory function for creating detector backend instance."""
    name = backend_name.lower().strip()
    if name in {"none", ""}:
        return NoneObjectDetector()
    elif name in {"yolo26-pytorch", "pytorch"}:
        path = model_path if model_path else "yolo26n.pt"
        return PyTorchYolo26Detector(
            model_path=path,
            confidence_threshold=confidence_threshold,
        )
    elif name in {"yolo26-onnx", "onnx"}:
        if not model_path:
            raise ValueError("model_path is required for yolo26-onnx backend")
        return ONNXYolo26Detector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
        )
    else:
        raise ValueError(f"Unsupported detector backend: {backend_name}")
