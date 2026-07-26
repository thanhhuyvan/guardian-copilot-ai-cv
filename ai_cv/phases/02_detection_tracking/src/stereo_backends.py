"""Backend-neutral stereo inference contract for the Phase 2B latency gate.

The learned adapters deliberately import their runtimes only when constructed.
This keeps the classical Stage 2A environment usable without PyTorch, ONNX
Runtime, or TensorRT installed.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import math
import subprocess
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import cv2
import numpy as np

from analyze_stereo_confidence import (
    compute_cropped_disparities_with_timing,
    configure_opencv_threads,
    create_left_matcher,
    create_right_matcher,
    left_right_consistency,
)


OPENSTEREO_REVISION = "23d71c92e33ad1f80dfc42bf29f5c6a914d38769"
LIGHTSTEREO_CHECKPOINT_SHA256 = (
    "3d768e0344c2b8bfacb8f7f27cc647cd338e5ba93ec66d944a9a73fd63ec9b2a"
)
LIGHTSTEREO_CONFIG_RELATIVE = Path(
    "cfgs/lightstereo/lightstereo_s_kitti.yaml"
)
LIGHTSTEREO_INPUT_SHAPE = (1, 3, 384, 640)
OPENSTEREO_ONNX_INPUT_NAMES = ("left_img", "right_img")
OPENSTEREO_ONNX_OUTPUT_NAME = "disp_pred"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BackendConfigurationError(RuntimeError):
    """Raised when an inference backend cannot be configured safely."""


class OptionalDependencyError(BackendConfigurationError):
    """Raised when a selected optional backend runtime is unavailable."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum for an artifact without loading it at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_lightstereo_checkpoint(path: Path) -> str:
    """Require the checksum-locked official LightStereo-S KITTI checkpoint."""
    resolved = _require_artifact(path, "LightStereo-S KITTI checkpoint")
    actual = sha256_file(resolved)
    if actual != LIGHTSTEREO_CHECKPOINT_SHA256:
        raise BackendConfigurationError(
            "LightStereo-S checkpoint SHA-256 mismatch: expected "
            f"{LIGHTSTEREO_CHECKPOINT_SHA256}, found {actual}. Fetch the "
            "checksum-locked official checkpoint; do not load this file."
        )
    return actual


def _immutable_float_mapping(values: Mapping[str, float]) -> dict[str, float]:
    result = {}
    for name, value in values.items():
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(f"timing {name!r} must be finite and non-negative")
        result[str(name)] = numeric
    return result


@dataclass(frozen=True)
class StereoResult:
    """One rectified left-view disparity result in native image coordinates."""

    disparity_px: np.ndarray
    valid_mask: np.ndarray
    confidence: np.ndarray | None
    backend: str
    precision: str
    input_shape: tuple[int, int, int, int]
    model_sha256: str | None
    timings_ms: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        disparity = np.ascontiguousarray(self.disparity_px, dtype=np.float32)
        valid = np.ascontiguousarray(self.valid_mask, dtype=bool)
        if disparity.ndim != 2:
            raise ValueError(
                f"disparity_px must have shape [H,W], got {disparity.shape}"
            )
        if valid.shape != disparity.shape:
            raise ValueError(
                "valid_mask must match disparity_px shape: "
                f"{valid.shape} != {disparity.shape}"
            )
        if not np.all(np.isfinite(disparity)):
            raise ValueError(
                "disparity_px must contain no NaN or Inf values, including "
                "pixels outside valid_mask"
            )
        if np.any(valid & (disparity <= 0)):
            raise ValueError(
                "valid_mask may only select finite, positive disparity pixels"
            )

        confidence = self.confidence
        if confidence is not None:
            confidence = np.ascontiguousarray(confidence, dtype=np.float32)
            if confidence.shape != disparity.shape:
                raise ValueError(
                    "confidence must match disparity_px shape: "
                    f"{confidence.shape} != {disparity.shape}"
                )
            finite_confidence = confidence[np.isfinite(confidence)]
            if finite_confidence.size and (
                np.min(finite_confidence) < 0 or np.max(finite_confidence) > 1
            ):
                raise ValueError("confidence values must be in the [0, 1] range")

        if len(self.input_shape) != 4 or any(
            int(dimension) <= 0 for dimension in self.input_shape
        ):
            raise ValueError("input_shape must be a positive NCHW shape")
        if not self.backend.strip():
            raise ValueError("backend must be non-empty")
        if not self.precision.strip():
            raise ValueError("precision must be non-empty")
        if self.model_sha256 is not None and (
            len(self.model_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.model_sha256)
        ):
            raise ValueError("model_sha256 must be a lowercase SHA-256 hex digest")

        object.__setattr__(self, "disparity_px", disparity)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(
            self, "input_shape", tuple(int(value) for value in self.input_shape)
        )
        object.__setattr__(
            self, "timings_ms", _immutable_float_mapping(self.timings_ms)
        )
        object.__setattr__(self, "metadata", dict(self.metadata))


def disparity_parity(
    reference: StereoResult, candidate: StereoResult
) -> dict[str, float | int]:
    """Compare a converted engine result with its FP32 disparity reference.

    Error percentiles use pixels valid in both results. Coverage changes are
    reported independently so a conversion cannot appear accurate merely by
    invalidating difficult pixels.
    """
    if reference.disparity_px.shape != candidate.disparity_px.shape:
        raise ValueError(
            "parity results must have identical native shapes: "
            f"{reference.disparity_px.shape} != {candidate.disparity_px.shape}"
        )
    shared = reference.valid_mask & candidate.valid_mask
    compared = int(np.count_nonzero(shared))
    reference_valid = int(np.count_nonzero(reference.valid_mask))
    if compared == 0:
        raise ValueError("parity comparison has no mutually valid pixels")
    absolute_error = np.abs(
        reference.disparity_px[shared] - candidate.disparity_px[shared]
    )
    missing = reference.valid_mask & ~candidate.valid_mask
    added = candidate.valid_mask & ~reference.valid_mask
    return {
        "compared_pixels": compared,
        "mean_absolute_error_px": float(np.mean(absolute_error)),
        "p95_absolute_error_px": float(np.percentile(absolute_error, 95)),
        "maximum_absolute_error_px": float(np.max(absolute_error)),
        "bad_1px_fraction": float(np.mean(absolute_error > 1.0)),
        "bad_3px_fraction": float(np.mean(absolute_error > 3.0)),
        "missing_reference_valid_fraction": float(
            np.count_nonzero(missing) / max(1, reference_valid)
        ),
        "additional_valid_fraction": float(
            np.count_nonzero(added) / max(1, reference.disparity_px.size)
        ),
    }


@runtime_checkable
class StereoBackend(Protocol):
    """Minimal protocol consumed by the end-to-end Guardian TTC benchmark."""

    name: str
    precision: str
    model_sha256: str | None

    def infer(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> StereoResult:
        """Infer native-resolution left disparity for one decoded stereo pair."""

    def close(self) -> None:
        """Release backend-owned resources."""

    def peak_gpu_memory_mb(self) -> float | None:
        """Return peak process GPU allocation when the runtime exposes it."""


def _validate_stereo_pair(
    left_bgr: np.ndarray, right_bgr: np.ndarray
) -> tuple[int, int]:
    if left_bgr.shape != right_bgr.shape:
        raise ValueError(
            f"left/right image shapes must match: {left_bgr.shape} != "
            f"{right_bgr.shape}"
        )
    if left_bgr.ndim != 3 or left_bgr.shape[2] != 3:
        raise ValueError(
            f"stereo images must have shape [H,W,3], got {left_bgr.shape}"
        )
    return int(left_bgr.shape[0]), int(left_bgr.shape[1])


class SgbmBackend:
    """Stage 2A SGBM reference exposed through the common stereo contract."""

    name = "sgbm"
    precision = "fp32"
    model_sha256 = None

    def __init__(
        self,
        *,
        opencv_threads: int = 6,
        stereo_workers: int = 1,
        stereo_roi_top: int = 0,
    ) -> None:
        if opencv_threads <= 0:
            raise BackendConfigurationError("opencv_threads must be positive")
        if stereo_workers not in (1, 2):
            raise BackendConfigurationError("stereo_workers must be 1 or 2")
        if stereo_roi_top < 0:
            raise BackendConfigurationError(
                "stereo_roi_top must be non-negative"
            )
        configure_opencv_threads(opencv_threads)
        self.opencv_threads = int(opencv_threads)
        self.stereo_workers = int(stereo_workers)
        self.stereo_roi_top = int(stereo_roi_top)
        self._left_matcher = create_left_matcher()
        self._right_matcher = create_right_matcher()
        self._executor = (
            ThreadPoolExecutor(max_workers=2, thread_name_prefix="sgbm")
            if stereo_workers == 2
            else None
        )

    def infer(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> StereoResult:
        height, width = _validate_stereo_pair(left_bgr, right_bgr)
        started = time.perf_counter()
        left, right, left_ms, right_ms = compute_cropped_disparities_with_timing(
            left_bgr,
            right_bgr,
            self._left_matcher,
            self._right_matcher,
            roi_top=self.stereo_roi_top,
            executor=self._executor,
        )
        consistency_started = time.perf_counter()
        valid, consistent, _ = left_right_consistency(left, right)
        consistency_ms = (time.perf_counter() - consistency_started) * 1000.0
        total_ms = (time.perf_counter() - started) * 1000.0
        matcher_wall_estimate = (
            left_ms + right_ms
            if self.stereo_workers == 1
            else max(left_ms, right_ms)
        )
        preprocess_orchestration_ms = max(
            0.0, total_ms - matcher_wall_estimate - consistency_ms
        )
        return StereoResult(
            disparity_px=left,
            valid_mask=valid,
            confidence=consistent.astype(np.float32),
            backend=self.name,
            precision=self.precision,
            input_shape=(1, 3, height, width),
            model_sha256=None,
            timings_ms={
                "preprocess_orchestration": preprocess_orchestration_ms,
                "left_match": left_ms,
                "right_match": right_ms,
                "consistency": consistency_ms,
                "stereo_total": total_ms,
            },
            metadata={
                "opencv_threads": self.opencv_threads,
                "stereo_workers": self.stereo_workers,
                "stereo_roi_top": self.stereo_roi_top,
            },
        )

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def peak_gpu_memory_mb(self) -> float:
        return 0.0


@dataclass(frozen=True)
class PadSpec:
    top: int
    right: int
    bottom: int
    left: int
    original_height: int
    original_width: int


class LightStereoPreprocessor:
    """Official OpenStereo RGB/ImageNet normalization and right-top padding.

    This mirrors the pinned checkout's
    ``stereo/datasets/dataset_utils/stereo_trans.py::RightTopPad``: padding is
    placed above and to the right with NumPy ``edge`` replication. The KITTI
    config then transposes to CHW and applies ImageNet normalization.
    """

    def __init__(
        self,
        target_shape: tuple[int, int] = LIGHTSTEREO_INPUT_SHAPE[2:],
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
    ) -> None:
        self.target_height, self.target_width = target_shape
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        if self.target_height <= 0 or self.target_width <= 0:
            raise ValueError("target_shape must be positive")
        if np.any(self.std <= 0):
            raise ValueError("normalization standard deviations must be positive")

    def prepare(
        self, left_bgr: np.ndarray, right_bgr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, PadSpec]:
        height, width = _validate_stereo_pair(left_bgr, right_bgr)
        if height > self.target_height or width > self.target_width:
            raise BackendConfigurationError(
                "LightStereo input exceeds the fixed engine shape: "
                f"{height}x{width} > {self.target_height}x{self.target_width}. "
                "Do not resize silently; export a matching fixed-shape engine."
            )

        pad = PadSpec(
            top=self.target_height - height,
            right=self.target_width - width,
            bottom=0,
            left=0,
            original_height=height,
            original_width=width,
        )
        pad_width = ((pad.top, 0), (0, pad.right), (0, 0))
        left_rgb = np.ascontiguousarray(left_bgr[..., ::-1], dtype=np.float32)
        right_rgb = np.ascontiguousarray(right_bgr[..., ::-1], dtype=np.float32)
        left_rgb = np.pad(left_rgb, pad_width, mode="edge")
        right_rgb = np.pad(right_rgb, pad_width, mode="edge")
        left_normalized = (left_rgb / 255.0 - self.mean) / self.std
        right_normalized = (right_rgb / 255.0 - self.mean) / self.std
        left_nchw = np.ascontiguousarray(
            left_normalized.transpose(2, 0, 1)[None], dtype=np.float32
        )
        right_nchw = np.ascontiguousarray(
            right_normalized.transpose(2, 0, 1)[None], dtype=np.float32
        )
        return left_nchw, right_nchw, pad

    def restore_disparity(self, output: Any, pad: PadSpec) -> np.ndarray:
        disparity = np.asarray(output)
        while disparity.ndim > 2 and disparity.shape[0] == 1:
            disparity = disparity[0]
        if disparity.shape != (self.target_height, self.target_width):
            raise BackendConfigurationError(
                "LightStereo output must reduce to the fixed padded shape "
                f"{self.target_height}x{self.target_width}; got {disparity.shape}"
            )
        y0 = pad.top
        y1 = y0 + pad.original_height
        x0 = pad.left
        x1 = x0 + pad.original_width
        restored = np.ascontiguousarray(disparity[y0:y1, x0:x1], dtype=np.float32)
        if restored.shape != (pad.original_height, pad.original_width):
            raise BackendConfigurationError(
                f"restored disparity has unexpected shape {restored.shape}"
            )
        return restored


def _optional_import(module_name: str, install_hint: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except (ImportError, OSError) as error:
        raise OptionalDependencyError(
            f"{module_name!r} is required for the selected backend. "
            f"{install_hint}. Original error: {error}"
        ) from error


def _require_artifact(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise BackendConfigurationError(
            f"{description} not found: {resolved}. Generate/download it in the "
            "external OpenStereo workspace; model artifacts must not be committed."
        )
    return resolved


def validate_lightstereo_onnx_model(
    model: Any, *, expected_precision: str | None = None
) -> None:
    """Require the pinned static opset-17 interchange contract."""
    default_opsets = [
        int(item.version)
        for item in model.opset_import
        if getattr(item, "domain", "") in ("", "ai.onnx")
    ]
    if default_opsets != [17]:
        raise BackendConfigurationError(
            f"LightStereo ONNX must use exactly opset 17; found {default_opsets}"
        )
    graph_inputs = {item.name: item for item in model.graph.input}
    if set(graph_inputs) != set(OPENSTEREO_ONNX_INPUT_NAMES):
        raise BackendConfigurationError(
            "LightStereo ONNX graph inputs must be exactly "
            f"{OPENSTEREO_ONNX_INPUT_NAMES}; found {sorted(graph_inputs)}"
        )
    for name in OPENSTEREO_ONNX_INPUT_NAMES:
        element_type = int(graph_inputs[name].type.tensor_type.elem_type)
        expected_element_type = {
            "fp32": 1,  # onnx.TensorProto.FLOAT
            "fp16": 10,  # onnx.TensorProto.FLOAT16
        }.get(expected_precision)
        if expected_element_type is not None and element_type != expected_element_type:
            raise BackendConfigurationError(
                f"{name} dtype does not match declared {expected_precision}: "
                f"ONNX TensorProto element type is {element_type}"
            )
        dimensions = tuple(
            int(dimension.dim_value)
            for dimension in graph_inputs[name].type.tensor_type.shape.dim
        )
        if dimensions != LIGHTSTEREO_INPUT_SHAPE:
            raise BackendConfigurationError(
                f"{name} must have static shape {LIGHTSTEREO_INPUT_SHAPE}; "
                f"found {dimensions}"
            )
    graph_outputs = {item.name: item for item in model.graph.output}
    if set(graph_outputs) != {OPENSTEREO_ONNX_OUTPUT_NAME}:
        raise BackendConfigurationError(
            "LightStereo ONNX graph output must be exactly "
            f"{OPENSTEREO_ONNX_OUTPUT_NAME!r}; found {sorted(graph_outputs)}"
        )
    output_shape = tuple(
        int(dimension.dim_value)
        for dimension in graph_outputs[
            OPENSTEREO_ONNX_OUTPUT_NAME
        ].type.tensor_type.shape.dim
    )
    allowed_output_shapes = {
        (1, LIGHTSTEREO_INPUT_SHAPE[2], LIGHTSTEREO_INPUT_SHAPE[3]),
        (1, 1, LIGHTSTEREO_INPUT_SHAPE[2], LIGHTSTEREO_INPUT_SHAPE[3]),
    }
    if output_shape not in allowed_output_shapes:
        raise BackendConfigurationError(
            "disp_pred must have a static native padded shape; found "
            f"{output_shape}"
        )


def verify_openstereo_revision(
    root: Path, expected_revision: str = OPENSTEREO_REVISION
) -> str:
    """Require the reviewed revision and a clean tracked OpenStereo tree."""
    resolved = root.expanduser().resolve()
    if not (resolved / ".git").is_dir() or not (resolved / "stereo").is_dir():
        raise BackendConfigurationError(
            f"OpenStereo checkout not found at {resolved}. Clone branch v2 into "
            "~/benchmarks/OpenStereo and check out "
            f"{expected_revision}."
        )
    try:
        revision_result = subprocess.run(
            ["git", "-C", str(resolved), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            [
                "git",
                "-C",
                str(resolved),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise BackendConfigurationError(
            f"cannot verify the OpenStereo revision at {resolved}: {error}"
        ) from error
    revision = revision_result.stdout.strip().lower()
    if revision != expected_revision:
        raise BackendConfigurationError(
            f"OpenStereo revision mismatch: expected {expected_revision}, got "
            f"{revision}. Run `git -C {resolved} checkout {expected_revision}`."
        )
    dirty = status_result.stdout.strip()
    if dirty:
        raise BackendConfigurationError(
            "OpenStereo tracked tree is dirty; refusing external code/config "
            f"changes at the pinned boundary: {dirty}"
        )
    return revision


def resolve_pinned_lightstereo_config(
    root: Path, config_path: Path | None = None
) -> Path:
    """Resolve only the reviewed LightStereo-S config inside the pinned tree."""
    resolved_root = root.expanduser().resolve()
    expected = (resolved_root / LIGHTSTEREO_CONFIG_RELATIVE).resolve()
    requested = (
        config_path.expanduser().resolve()
        if config_path is not None
        else expected
    )
    if requested != expected or not expected.is_relative_to(resolved_root):
        raise BackendConfigurationError(
            "LightStereo config must be the pinned in-tree path "
            f"{LIGHTSTEREO_CONFIG_RELATIVE.as_posix()}; got {requested}"
        )
    if not expected.is_file():
        raise BackendConfigurationError(
            f"pinned LightStereo-S config not found: {expected}"
        )
    relative = LIGHTSTEREO_CONFIG_RELATIVE.as_posix()
    try:
        tracked = subprocess.run(
            [
                "git",
                "-C",
                str(resolved_root),
                "ls-files",
                "--error-unmatch",
                "--",
                relative,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        expected_blob = subprocess.run(
            [
                "git",
                "-C",
                str(resolved_root),
                "rev-parse",
                f"HEAD:{relative}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        actual_blob = subprocess.run(
            ["git", "-C", str(resolved_root), "hash-object", str(expected)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise BackendConfigurationError(
            f"cannot verify pinned LightStereo config {expected}: {error}"
        ) from error
    if not tracked.stdout.strip():
        raise BackendConfigurationError(
            f"pinned LightStereo config is not tracked: {expected}"
        )
    if expected_blob.stdout.strip() != actual_blob.stdout.strip():
        raise BackendConfigurationError(
            "LightStereo config content differs from the pinned commit"
        )
    return expected


class _ConfigNode(dict[str, Any]):
    """Minimal recursive attribute mapping for an OpenStereo YAML config."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def _config_node(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _ConfigNode(
            {str(key): _config_node(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_config_node(item) for item in value]
    return value


def load_lightstereo_config(config_path: Path, yaml_module: Any) -> _ConfigNode:
    """Parse the pinned YAML without importing OpenStereo utility packages."""
    try:
        document = yaml_module.safe_load(
            config_path.read_text(encoding="utf-8")
        )
    except Exception as error:
        raise BackendConfigurationError(
            f"cannot parse LightStereo config {config_path}: {error}"
        ) from error
    configuration = _config_node(document)
    if not isinstance(configuration, _ConfigNode):
        raise BackendConfigurationError("LightStereo config must be a mapping")
    model = configuration.get("MODEL")
    if not isinstance(model, _ConfigNode) or model.get("NAME") != "LightStereo":
        raise BackendConfigurationError(
            "pinned config MODEL.NAME must be LightStereo"
        )
    return configuration


_LIGHTSTEREO_NAMESPACE_PATHS = {
    "stereo": Path("stereo"),
    "stereo.modeling": Path("stereo/modeling"),
    "stereo.modeling.common": Path("stereo/modeling/common"),
    "stereo.modeling.cost_volume": Path("stereo/modeling/cost_volume"),
    "stereo.modeling.disp_pred": Path("stereo/modeling/disp_pred"),
    "stereo.modeling.disp_refinement": Path("stereo/modeling/disp_refinement"),
    "stereo.modeling.models": Path("stereo/modeling/models"),
    "stereo.modeling.models.lightstereo": Path(
        "stereo/modeling/models/lightstereo"
    ),
}
_LIGHTSTEREO_MODULE_PATHS = (
    (
        "stereo.modeling.common.basic_block_2d",
        Path("stereo/modeling/common/basic_block_2d.py"),
    ),
    (
        "stereo.modeling.common.basic_block_3d",
        Path("stereo/modeling/common/basic_block_3d.py"),
    ),
    (
        "stereo.modeling.cost_volume.cost_volume",
        Path("stereo/modeling/cost_volume/cost_volume.py"),
    ),
    (
        "stereo.modeling.disp_pred.disp_regression",
        Path("stereo/modeling/disp_pred/disp_regression.py"),
    ),
    (
        "stereo.modeling.disp_refinement.disp_refinement",
        Path("stereo/modeling/disp_refinement/disp_refinement.py"),
    ),
    (
        "stereo.modeling.models.lightstereo.aggregation",
        Path("stereo/modeling/models/lightstereo/aggregation.py"),
    ),
    (
        "stereo.modeling.models.lightstereo.backbone",
        Path("stereo/modeling/models/lightstereo/backbone.py"),
    ),
    (
        "stereo.modeling.models.lightstereo.lightstereo",
        Path("stereo/modeling/models/lightstereo/lightstereo.py"),
    ),
)


def _namespace_module(name: str, directory: Path, source_root: Path) -> Any:
    module = types.ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(directory)]
    module.__guardian_lightstereo_only__ = True
    module.__guardian_source_root__ = str(source_root)
    spec = importlib.util.spec_from_loader(name, loader=None, is_package=True)
    if spec is not None:
        spec.submodule_search_locations = [str(directory)]
    module.__spec__ = spec
    return module


def _attach_to_parent(name: str, module: Any) -> None:
    parent_name, _, child_name = name.rpartition(".")
    if parent_name:
        parent = sys.modules[parent_name]
        setattr(parent, child_name, module)


def load_pinned_lightstereo_class(openstereo_root: Path) -> Any:
    """Load only LightStereo source files without modeling/__init__.py.

    Synthetic namespace packages prevent Python from executing OpenStereo's
    broad model registry, which imports optional FoundationStereo/flash-attn.
    """
    root = openstereo_root.expanduser().resolve()
    target_name = "stereo.modeling.models.lightstereo.lightstereo"
    existing_target = sys.modules.get(target_name)
    if existing_target is not None:
        if (
            getattr(existing_target, "__guardian_source_root__", None)
            != str(root)
        ):
            raise BackendConfigurationError(
                "a LightStereo module from a different source root is loaded"
            )
        return existing_target.LightStereo

    conflicts = [
        name
        for name, module in sys.modules.items()
        if (name == "stereo" or name.startswith("stereo."))
        and not getattr(module, "__guardian_lightstereo_only__", False)
    ]
    if conflicts:
        raise BackendConfigurationError(
            "OpenStereo modules were imported before the safe LightStereo-only "
            f"loader ({sorted(conflicts)[:5]}). Start a clean Python process."
        )

    created: list[str] = []
    try:
        for name, relative in _LIGHTSTEREO_NAMESPACE_PATHS.items():
            directory = (root / relative).resolve()
            if not directory.is_relative_to(root) or not directory.is_dir():
                raise BackendConfigurationError(
                    f"pinned LightStereo package directory missing: {directory}"
                )
            if name not in sys.modules:
                module = _namespace_module(name, directory, root)
                sys.modules[name] = module
                created.append(name)
                _attach_to_parent(name, module)

        for name, relative in _LIGHTSTEREO_MODULE_PATHS:
            source = (root / relative).resolve()
            if not source.is_relative_to(root) or not source.is_file():
                raise BackendConfigurationError(
                    f"pinned LightStereo source file missing: {source}"
                )
            spec = importlib.util.spec_from_file_location(name, source)
            if spec is None or spec.loader is None:
                raise BackendConfigurationError(
                    f"cannot create import spec for {source}"
                )
            module = importlib.util.module_from_spec(spec)
            module.__guardian_lightstereo_only__ = True
            module.__guardian_source_root__ = str(root)
            sys.modules[name] = module
            created.append(name)
            _attach_to_parent(name, module)
            spec.loader.exec_module(module)
        return sys.modules[target_name].LightStereo
    except Exception as error:
        for name in reversed(created):
            sys.modules.pop(name, None)
        if isinstance(error, BackendConfigurationError):
            raise
        raise OptionalDependencyError(
            "LightStereo-only imports failed. Install only its PyTorch, timm, "
            f"and OpenStereo support dependencies: {error}"
        ) from error


def instantiate_pinned_lightstereo(
    *,
    torch_module: Any,
    yaml_module: Any,
    timm_module: Any,
    checkpoint_path: Path,
    openstereo_root: Path,
    config_path: Path | None,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    """Build and load the checksum-locked model without unsafe pickle code."""
    root = openstereo_root.expanduser().resolve()
    revision = verify_openstereo_revision(root)
    config = resolve_pinned_lightstereo_config(root, config_path)
    checkpoint = checkpoint_path.expanduser().resolve()
    checkpoint_sha256 = verify_lightstereo_checkpoint(checkpoint)
    configuration = load_lightstereo_config(config, yaml_module)
    lightstereo_class = load_pinned_lightstereo_class(root)

    original_create_model = timm_module.create_model

    def create_model_without_download(
        model_name: str, *args: Any, **kwargs: Any
    ) -> Any:
        kwargs["pretrained"] = False
        return original_create_model(model_name, *args, **kwargs)

    timm_module.create_model = create_model_without_download
    try:
        model = lightstereo_class(configuration.MODEL)
    finally:
        timm_module.create_model = original_create_model

    try:
        checkpoint_document = torch_module.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise BackendConfigurationError(
            f"safe LightStereo checkpoint loading failed: {error}"
        ) from error
    if not isinstance(checkpoint_document, Mapping):
        raise BackendConfigurationError(
            "LightStereo checkpoint must contain a state-dict mapping"
        )
    state = checkpoint_document.get(
        "model_state",
        checkpoint_document.get("model", checkpoint_document),
    )
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) for key in state
    ):
        raise BackendConfigurationError(
            "LightStereo checkpoint state must map string names to tensors"
        )
    normalized_state = {
        key.replace("module.", "", 1) if key.startswith("module.") else key: value
        for key, value in state.items()
    }
    model.load_state_dict(normalized_state, strict=True)
    model.eval().to(device)
    provenance = {
        "openstereo_revision": revision,
        "openstereo_tracked_tree_clean": True,
        "checkpoint": {
            "name": checkpoint.name,
            "sha256": checkpoint_sha256,
            "expected_sha256": LIGHTSTEREO_CHECKPOINT_SHA256,
        },
        "config": {
            "relative_path": LIGHTSTEREO_CONFIG_RELATIVE.as_posix(),
            "sha256": sha256_file(config),
        },
        "safe_checkpoint_load": {"weights_only": True, "map_location": "cpu"},
        "model_import_route": "guardian-lightstereo-only",
        "backbone_pretrained_download": False,
    }
    return model, provenance


class _LightStereoBackendBase:
    input_shape = LIGHTSTEREO_INPUT_SHAPE

    def __init__(self, *, name: str, precision: str, artifact_path: Path) -> None:
        self.name = name
        self.precision = precision
        self.artifact_path = _require_artifact(artifact_path, "model artifact")
        self.model_sha256 = sha256_file(self.artifact_path)
        self.preprocessor = LightStereoPreprocessor()

    def _make_result(
        self,
        raw_output: Any,
        pad: PadSpec,
        started: float,
        timings: Mapping[str, float],
        metadata: Mapping[str, Any] | None = None,
    ) -> StereoResult:
        postprocess_started = time.perf_counter()
        disparity = self.preprocessor.restore_disparity(raw_output, pad)
        valid = np.isfinite(disparity) & (disparity > 0.5)
        postprocess_ms = (time.perf_counter() - postprocess_started) * 1000.0
        all_timings = dict(timings)
        all_timings["postprocess"] = postprocess_ms
        all_timings["stereo_total"] = (time.perf_counter() - started) * 1000.0
        return StereoResult(
            disparity_px=disparity,
            valid_mask=valid,
            confidence=None,
            backend=self.name,
            precision=self.precision,
            input_shape=self.input_shape,
            model_sha256=self.model_sha256,
            timings_ms=all_timings,
            metadata=metadata or {},
        )

    def close(self) -> None:
        return None


class LightStereoPyTorchBackend(_LightStereoBackendBase):
    """OpenStereo LightStereo-S checkpoint runner used as the FP32 reference."""

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        openstereo_root: Path,
        config_path: Path | None = None,
        precision: str = "fp32",
        device: str = "cuda:0",
    ) -> None:
        if precision not in {"fp32", "fp16"}:
            raise BackendConfigurationError(
                "lightstereo-pytorch precision must be fp32 or fp16"
            )
        super().__init__(
            name="lightstereo-pytorch",
            precision=precision,
            artifact_path=checkpoint_path,
        )
        self.openstereo_root = openstereo_root.expanduser().resolve()
        self._torch = _optional_import(
            "torch",
            "install the CUDA 12.8 PyTorch wheel in the OpenStereo environment",
        )
        yaml = _optional_import(
            "yaml", "install OpenStereo requirements (`python -m pip install pyyaml`)"
        )
        timm = _optional_import(
            "timm", "install the pinned OpenStereo timm dependency"
        )
        if not self._torch.cuda.is_available():
            raise BackendConfigurationError(
                "PyTorch cannot see CUDA. In WSL, verify `nvidia-smi`, then install "
                "the CUDA-enabled PyTorch wheel instead of the CPU-only wheel."
            )
        self.device = self._torch.device(device)
        self._model, self.source_provenance = instantiate_pinned_lightstereo(
            torch_module=self._torch,
            yaml_module=yaml,
            timm_module=timm,
            checkpoint_path=self.artifact_path,
            openstereo_root=self.openstereo_root,
            config_path=config_path,
            device=self.device,
        )
        self.openstereo_revision = self.source_provenance[
            "openstereo_revision"
        ]
        self.config_path = resolve_pinned_lightstereo_config(
            self.openstereo_root, config_path
        )
        self._torch.cuda.reset_peak_memory_stats(self.device)

    def infer(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> StereoResult:
        started = time.perf_counter()
        preprocess_started = time.perf_counter()
        left, right, pad = self.preprocessor.prepare(left_bgr, right_bgr)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0

        self._torch.cuda.synchronize(self.device)
        inference_started = time.perf_counter()
        with self._torch.inference_mode():
            left_tensor = self._torch.from_numpy(left).to(self.device)
            right_tensor = self._torch.from_numpy(right).to(self.device)
            with self._torch.autocast(
                device_type="cuda",
                dtype=self._torch.float16,
                enabled=self.precision == "fp16",
            ):
                output = self._model({"left": left_tensor, "right": right_tensor})
            disparity = output["disp_pred"].float().cpu().numpy()
        self._torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter() - inference_started) * 1000.0
        return self._make_result(
            disparity,
            pad,
            started,
            {"preprocess": preprocess_ms, "inference": inference_ms},
            {
                "openstereo_revision": self.openstereo_revision,
                "config_path": str(self.config_path),
                "device": str(self.device),
                "source_provenance": self.source_provenance,
            },
        )

    def peak_gpu_memory_mb(self) -> float:
        return float(
            self._torch.cuda.max_memory_reserved(self.device) / (1024 * 1024)
        )


class LightStereoOnnxBackend(_LightStereoBackendBase):
    """Static-shape ONNX Runtime CUDA adapter for the official OpenStereo export."""

    def __init__(
        self,
        *,
        model_path: Path,
        precision: str = "fp32",
        device_id: int = 0,
    ) -> None:
        if precision not in {"fp32", "fp16"}:
            raise BackendConfigurationError(
                "lightstereo-onnx precision must be fp32 or fp16"
            )
        super().__init__(
            name="lightstereo-onnx",
            precision=precision,
            artifact_path=model_path,
        )
        self._ort = _optional_import(
            "onnxruntime",
            "install `onnxruntime-gpu` in the dedicated OpenStereo environment",
        )
        onnx = _optional_import(
            "onnx",
            "install `onnx` in the dedicated OpenStereo environment",
        )
        try:
            onnx_model = onnx.load(str(self.artifact_path))
            onnx.checker.check_model(onnx_model)
            validate_lightstereo_onnx_model(
                onnx_model, expected_precision=precision
            )
        except Exception as error:
            if isinstance(error, BackendConfigurationError):
                raise
            raise BackendConfigurationError(
                f"ONNX checker rejected {self.artifact_path}: {error}"
            ) from error
        available = set(self._ort.get_available_providers())
        if "CUDAExecutionProvider" not in available:
            raise BackendConfigurationError(
                "ONNX Runtime CUDAExecutionProvider is unavailable. Remove the CPU "
                "`onnxruntime` package and install a CUDA-compatible "
                "`onnxruntime-gpu` build."
            )
        try:
            self._session = self._ort.InferenceSession(
                str(self.artifact_path),
                providers=[
                    ("CUDAExecutionProvider", {"device_id": int(device_id)})
                ],
            )
        except Exception as error:
            raise BackendConfigurationError(
                f"failed to load ONNX model with CUDAExecutionProvider: {error}"
            ) from error
        active = self._session.get_providers()
        if not active or active[0] != "CUDAExecutionProvider":
            raise BackendConfigurationError(
                f"ONNX Runtime fell back from CUDA; active providers are {active}"
            )
        inputs = {item.name: item for item in self._session.get_inputs()}
        if set(inputs) != set(OPENSTEREO_ONNX_INPUT_NAMES):
            raise BackendConfigurationError(
                "official OpenStereo ONNX inputs must be exactly left_img and "
                f"right_img; found {sorted(inputs)}"
            )
        outputs = {item.name: item for item in self._session.get_outputs()}
        if set(outputs) != {OPENSTEREO_ONNX_OUTPUT_NAME}:
            raise BackendConfigurationError(
                "official OpenStereo ONNX output must be exactly disp_pred; "
                f"found {sorted(outputs)}"
            )
        self._input_types = {
            name: np.float16 if item.type == "tensor(float16)" else np.float32
            for name, item in inputs.items()
        }
        expected_ort_type = (
            "tensor(float16)" if precision == "fp16" else "tensor(float)"
        )
        mismatched_types = {
            name: item.type
            for name, item in inputs.items()
            if item.type != expected_ort_type
        }
        if mismatched_types:
            raise BackendConfigurationError(
                "ONNX Runtime input dtype does not match declared "
                f"{precision}: {mismatched_types}"
            )
        self.device_id = int(device_id)

    def infer(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> StereoResult:
        started = time.perf_counter()
        preprocess_started = time.perf_counter()
        left, right, pad = self.preprocessor.prepare(left_bgr, right_bgr)
        left = left.astype(self._input_types["left_img"], copy=False)
        right = right.astype(self._input_types["right_img"], copy=False)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0
        inference_started = time.perf_counter()
        disparity = self._session.run(
            [OPENSTEREO_ONNX_OUTPUT_NAME],
            {
                OPENSTEREO_ONNX_INPUT_NAMES[0]: left,
                OPENSTEREO_ONNX_INPUT_NAMES[1]: right,
            },
        )[0]
        inference_ms = (time.perf_counter() - inference_started) * 1000.0
        return self._make_result(
            disparity,
            pad,
            started,
            {"preprocess": preprocess_ms, "inference": inference_ms},
            {
                "providers": self._session.get_providers(),
                "device_id": self.device_id,
            },
        )

    def peak_gpu_memory_mb(self) -> None:
        # ORT does not expose process peak VRAM through its stable Python API.
        return None


class LightStereoTensorRtBackend(_LightStereoBackendBase):
    """TensorRT 10 engine runner using PyTorch CUDA tensors for device buffers."""

    def __init__(
        self,
        *,
        engine_path: Path,
        precision: str,
        device_id: int = 0,
    ) -> None:
        if precision not in {"fp32", "fp16", "int8"}:
            raise BackendConfigurationError(
                "lightstereo-tensorrt precision must be fp32, fp16, or int8"
            )
        super().__init__(
            name="lightstereo-tensorrt",
            precision=precision,
            artifact_path=engine_path,
        )
        self._trt = _optional_import(
            "tensorrt",
            "install TensorRT 10.x matching CUDA 12.8 inside WSL",
        )
        self._torch = _optional_import(
            "torch",
            "install the CUDA 12.8 PyTorch wheel for TensorRT device buffers",
        )
        if not self._torch.cuda.is_available():
            raise BackendConfigurationError(
                "TensorRT backend requires CUDA-visible PyTorch device buffers"
            )
        required_api = (
            "num_io_tensors",
            "get_tensor_name",
            "get_tensor_mode",
            "get_tensor_shape",
        )
        self.device = self._torch.device(f"cuda:{int(device_id)}")
        self._logger = self._trt.Logger(self._trt.Logger.WARNING)
        self._runtime = self._trt.Runtime(self._logger)
        self._engine = self._runtime.deserialize_cuda_engine(
            self.artifact_path.read_bytes()
        )
        if self._engine is None:
            raise BackendConfigurationError(
                f"TensorRT could not deserialize engine {self.artifact_path}"
            )
        if any(not hasattr(self._engine, attribute) for attribute in required_api):
            raise BackendConfigurationError(
                "this adapter requires the TensorRT 10 named-tensor Python API"
            )
        self._context = self._engine.create_execution_context()
        if self._context is None or not hasattr(self._context, "execute_async_v3"):
            raise BackendConfigurationError(
                "TensorRT execution context does not support execute_async_v3"
            )
        self._tensor_names = [
            self._engine.get_tensor_name(index)
            for index in range(self._engine.num_io_tensors)
        ]
        input_names = {
            name
            for name in self._tensor_names
            if self._engine.get_tensor_mode(name) == self._trt.TensorIOMode.INPUT
        }
        output_names = set(self._tensor_names) - input_names
        if (
            input_names != set(OPENSTEREO_ONNX_INPUT_NAMES)
            or output_names != {OPENSTEREO_ONNX_OUTPUT_NAME}
        ):
            raise BackendConfigurationError(
                "TensorRT engine must expose left_img, right_img -> disp_pred; "
                f"found inputs={sorted(input_names)}, outputs={sorted(output_names)}"
            )
        for name in OPENSTEREO_ONNX_INPUT_NAMES:
            shape = tuple(int(value) for value in self._engine.get_tensor_shape(name))
            if shape != LIGHTSTEREO_INPUT_SHAPE:
                raise BackendConfigurationError(
                    f"TensorRT {name} must have fixed shape "
                    f"{LIGHTSTEREO_INPUT_SHAPE}; found {shape}"
                )
        output_shape = tuple(
            int(value)
            for value in self._engine.get_tensor_shape(
                OPENSTEREO_ONNX_OUTPUT_NAME
            )
        )
        if output_shape not in {
            (1, LIGHTSTEREO_INPUT_SHAPE[2], LIGHTSTEREO_INPUT_SHAPE[3]),
            (1, 1, LIGHTSTEREO_INPUT_SHAPE[2], LIGHTSTEREO_INPUT_SHAPE[3]),
        }:
            raise BackendConfigurationError(
                "TensorRT disp_pred must have a fixed 384x640 output; "
                f"found {output_shape}"
            )
        self._torch.cuda.reset_peak_memory_stats(self.device)

    def _torch_dtype(self, tensor_name: str) -> Any:
        numpy_dtype = np.dtype(
            self._trt.nptype(self._engine.get_tensor_dtype(tensor_name))
        )
        mapping = {
            np.dtype(np.float16): self._torch.float16,
            np.dtype(np.float32): self._torch.float32,
            np.dtype(np.int8): self._torch.int8,
            np.dtype(np.int32): self._torch.int32,
        }
        if numpy_dtype not in mapping:
            raise BackendConfigurationError(
                f"unsupported TensorRT tensor dtype {numpy_dtype} for {tensor_name}"
            )
        return mapping[numpy_dtype]

    def infer(self, left_bgr: np.ndarray, right_bgr: np.ndarray) -> StereoResult:
        started = time.perf_counter()
        preprocess_started = time.perf_counter()
        left, right, pad = self.preprocessor.prepare(left_bgr, right_bgr)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0

        self._torch.cuda.synchronize(self.device)
        inference_started = time.perf_counter()
        host_inputs = {
            OPENSTEREO_ONNX_INPUT_NAMES[0]: left,
            OPENSTEREO_ONNX_INPUT_NAMES[1]: right,
        }
        tensors = {}
        for name, host_value in host_inputs.items():
            shape = tuple(int(value) for value in host_value.shape)
            if not self._context.set_input_shape(name, shape):
                raise BackendConfigurationError(
                    f"TensorRT rejected input shape {shape} for {name}"
                )
            tensors[name] = self._torch.from_numpy(host_value).to(
                device=self.device, dtype=self._torch_dtype(name)
            )

        for name in self._tensor_names:
            if name in tensors:
                continue
            shape = tuple(int(value) for value in self._context.get_tensor_shape(name))
            if any(value <= 0 for value in shape):
                raise BackendConfigurationError(
                    f"unresolved TensorRT output shape {shape} for {name}"
                )
            tensors[name] = self._torch.empty(
                shape, device=self.device, dtype=self._torch_dtype(name)
            )
        for name, tensor in tensors.items():
            if not self._context.set_tensor_address(name, int(tensor.data_ptr())):
                raise BackendConfigurationError(
                    f"TensorRT rejected the device buffer for {name}"
                )
        stream = self._torch.cuda.current_stream(self.device)
        if not self._context.execute_async_v3(stream.cuda_stream):
            raise BackendConfigurationError("TensorRT execute_async_v3 failed")
        stream.synchronize()
        disparity = tensors[OPENSTEREO_ONNX_OUTPUT_NAME].float().cpu().numpy()
        self._torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter() - inference_started) * 1000.0
        return self._make_result(
            disparity,
            pad,
            started,
            {"preprocess": preprocess_ms, "inference": inference_ms},
            {
                "tensorrt_version": self._trt.__version__,
                "device": str(self.device),
            },
        )

    def peak_gpu_memory_mb(self) -> float | None:
        # TensorRT owns engine/workspace allocations outside PyTorch's allocator.
        # The benchmark's process-wide NVML sampler is the authoritative source.
        return None


def create_backend(
    backend: str,
    *,
    precision: str,
    model_path: Path | None = None,
    openstereo_root: Path | None = None,
    config_path: Path | None = None,
    device_id: int = 0,
    opencv_threads: int = 6,
    stereo_workers: int = 1,
    stereo_roi_top: int = 0,
) -> StereoBackend:
    """Construct a selected backend with all dependency checks deferred."""
    if backend == "sgbm":
        if precision != "fp32":
            raise BackendConfigurationError("sgbm only supports --precision fp32")
        return SgbmBackend(
            opencv_threads=opencv_threads,
            stereo_workers=stereo_workers,
            stereo_roi_top=stereo_roi_top,
        )
    if model_path is None:
        raise BackendConfigurationError(
            f"--model-path is required for backend {backend}"
        )
    if backend == "lightstereo-pytorch":
        if openstereo_root is None:
            raise BackendConfigurationError(
                "--openstereo-root is required for lightstereo-pytorch"
            )
        return LightStereoPyTorchBackend(
            checkpoint_path=model_path,
            openstereo_root=openstereo_root,
            config_path=config_path,
            precision=precision,
            device=f"cuda:{device_id}",
        )
    if backend == "lightstereo-onnx":
        return LightStereoOnnxBackend(
            model_path=model_path, precision=precision, device_id=device_id
        )
    if backend == "lightstereo-tensorrt":
        return LightStereoTensorRtBackend(
            engine_path=model_path, precision=precision, device_id=device_id
        )
    raise BackendConfigurationError(f"unknown backend {backend!r}")
