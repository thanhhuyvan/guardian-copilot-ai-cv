"""Reproducible OpenStereo LightStereo-S export and TensorRT build tools.

This module deliberately keeps PyTorch, ONNX, and TensorRT as optional runtime
dependencies.  Dataset-manifest generation and its unit tests therefore work
in the classical Guardian environment without a GPU.

The external OpenStereo checkout and every generated model artifact must live
outside tracked source.  See ``notes/PHASE02B_LEARNED_DEPLOYMENT.md``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from stereo_backends import (
    LIGHTSTEREO_CONFIG_RELATIVE,
    LIGHTSTEREO_CHECKPOINT_SHA256,
    LIGHTSTEREO_INPUT_SHAPE,
    OPENSTEREO_ONNX_INPUT_NAMES,
    OPENSTEREO_ONNX_OUTPUT_NAME,
    OPENSTEREO_REVISION,
    BackendConfigurationError,
    OptionalDependencyError,
    instantiate_pinned_lightstereo,
    sha256_file,
    resolve_pinned_lightstereo_config,
    verify_lightstereo_checkpoint,
    validate_lightstereo_onnx_model,
    verify_openstereo_revision,
)


TRIPS = tuple(f"T0{index}-Sample" for index in range(1, 7))
PAIR_MANIFEST_SCHEMA = "guardian.phase02b.stereo-pairs.v1"
ARTIFACT_MANIFEST_SCHEMA = "guardian.phase02b.model-artifact.v1"
PARITY_KIND = "lightstereo-onnx-parity"
CALIBRATION_KIND = "lightstereo-tensorrt-int8-calibration"
PARITY_STRIDE = 50
PARITY_TOTAL = 72
CALIBRATION_PER_TRIP = 50
CALIBRATION_TOTAL = 300
INPUT_HEIGHT = LIGHTSTEREO_INPUT_SHAPE[2]
INPUT_WIDTH = LIGHTSTEREO_INPUT_SHAPE[3]
NATIVE_HEIGHT = 360
NATIVE_WIDTH = 640
TENSORRT_WORKSPACE_BYTES = 2 * 1024**3
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


class DeploymentManifestError(BackendConfigurationError):
    """Raised when a parity or calibration manifest is unsafe to consume."""


@dataclass(frozen=True, order=True)
class StereoPair:
    """One dataset-relative, rectified stereo image pair."""

    trip_id: str
    frame_id: int
    left: str
    right: str

    def __post_init__(self) -> None:
        if not self.trip_id:
            raise ValueError("trip_id must not be empty")
        if int(self.frame_id) < 0:
            raise ValueError("frame_id must be non-negative")
        for name, value in (("left", self.left), ("right", self.right)):
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{name} must be a safe dataset-relative path")

    def as_dict(self) -> dict[str, Any]:
        return {
            "trip_id": self.trip_id,
            "frame_id": int(self.frame_id),
            "left": Path(self.left).as_posix(),
            "right": Path(self.right).as_posix(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StereoPair":
        try:
            return cls(
                trip_id=str(value["trip_id"]),
                frame_id=int(value["frame_id"]),
                left=str(value["left"]),
                right=str(value["right"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise DeploymentManifestError(
                f"invalid stereo-pair entry: {value!r}"
            ) from error


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value independent of indentation or key order."""
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
            + "\n"
        )
    os.replace(temporary, path)


def _read_trip_frame_ids(trip_dir: Path) -> list[int]:
    metadata = trip_dir / f"{trip_dir.name}.json.gz"
    if not metadata.is_file():
        candidates = sorted(trip_dir.glob("*.json.gz"))
        if len(candidates) != 1:
            raise DeploymentManifestError(
                f"{trip_dir}: expected one trip metadata .json.gz, found "
                f"{len(candidates)}"
            )
        metadata = candidates[0]
    try:
        with gzip.open(metadata, "rt", encoding="utf-8") as handle:
            document = json.load(handle)
        frame_ids = [int(item["frame_id"]) for item in document["frames"]]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise DeploymentManifestError(
            f"cannot read frame ids from {metadata}: {error}"
        ) from error
    if not frame_ids or len(frame_ids) != len(set(frame_ids)):
        raise DeploymentManifestError(
            f"{metadata}: frame ids must be non-empty and unique"
        )
    if frame_ids != sorted(frame_ids):
        raise DeploymentManifestError(f"{metadata}: frame ids are not ordered")
    return frame_ids


def _image_relative_path(
    data_root: Path, trip_dir: Path, camera_directory: str, frame_id: int
) -> str:
    stem = f"{frame_id:06d}"
    directory = trip_dir / "kitti" / camera_directory
    matches = [
        directory / f"{stem}{extension}"
        for extension in IMAGE_EXTENSIONS
        if (directory / f"{stem}{extension}").is_file()
    ]
    if len(matches) != 1:
        raise DeploymentManifestError(
            f"{directory}: frame {frame_id} must have exactly one image among "
            f"{IMAGE_EXTENSIONS}; found {len(matches)}"
        )
    return matches[0].relative_to(data_root).as_posix()


def discover_stereo_pairs(
    data_root: Path, trip_ids: Sequence[str] = TRIPS
) -> dict[str, list[StereoPair]]:
    """Discover ordered frame pairs without importing the starter-kit loader."""
    root = data_root.expanduser().resolve()
    if not root.is_dir():
        raise DeploymentManifestError(f"practice dataset not found: {root}")
    discovered: dict[str, list[StereoPair]] = {}
    for trip_id in trip_ids:
        trip_dir = root / trip_id
        if not trip_dir.is_dir():
            raise DeploymentManifestError(f"trip directory not found: {trip_dir}")
        pairs = []
        for frame_id in _read_trip_frame_ids(trip_dir):
            pairs.append(
                StereoPair(
                    trip_id=trip_id,
                    frame_id=frame_id,
                    left=_image_relative_path(
                        root, trip_dir, "image_2", frame_id
                    ),
                    right=_image_relative_path(
                        root, trip_dir, "image_3", frame_id
                    ),
                )
            )
        discovered[trip_id] = pairs
    return discovered


def _evenly_spaced(items: Sequence[StereoPair], count: int) -> list[StereoPair]:
    """Select deterministic temporal-bin midpoints without endpoint bias."""
    if count <= 0:
        raise ValueError("selection count must be positive")
    if len(items) < count:
        raise DeploymentManifestError(
            f"cannot select {count} unique samples from {len(items)} candidates"
        )
    indices = [
        ((2 * index + 1) * len(items)) // (2 * count)
        for index in range(count)
    ]
    if len(indices) != len(set(indices)):
        raise DeploymentManifestError("temporal-bin selection produced duplicates")
    return [items[index] for index in indices]


def select_deployment_pairs(
    pairs_by_trip: Mapping[str, Sequence[StereoPair]],
    *,
    trip_ids: Sequence[str] = TRIPS,
    parity_stride: int = PARITY_STRIDE,
    calibration_per_trip: int = CALIBRATION_PER_TRIP,
) -> tuple[list[StereoPair], list[StereoPair]]:
    """Select parity and disjoint INT8 calibration samples deterministically."""
    if parity_stride <= 0:
        raise ValueError("parity_stride must be positive")
    parity: list[StereoPair] = []
    calibration_by_trip: dict[str, list[StereoPair]] = {}
    for trip_id in trip_ids:
        try:
            ordered = list(pairs_by_trip[trip_id])
        except KeyError as error:
            raise DeploymentManifestError(f"missing trip {trip_id}") from error
        if not ordered:
            raise DeploymentManifestError(f"trip {trip_id} has no stereo pairs")
        if ordered != sorted(ordered, key=lambda item: item.frame_id):
            raise DeploymentManifestError(f"trip {trip_id} pairs are not ordered")
        if any(item.trip_id != trip_id for item in ordered):
            raise DeploymentManifestError(
                f"trip {trip_id} contains a pair with a different trip_id"
            )
        selected_parity = ordered[::parity_stride]
        parity_keys = {
            (item.trip_id, item.frame_id) for item in selected_parity
        }
        eligible = [
            item
            for item in ordered
            if (item.trip_id, item.frame_id) not in parity_keys
        ]
        selected_calibration = _evenly_spaced(
            eligible, calibration_per_trip
        )
        parity.extend(selected_parity)
        calibration_by_trip[trip_id] = selected_calibration

    # Entropy calibration can be order-sensitive. Interleave temporal bins
    # across trips instead of presenting all 50 images from one trip at once.
    calibration = [
        calibration_by_trip[trip_id][sample_index]
        for sample_index in range(calibration_per_trip)
        for trip_id in trip_ids
    ]

    parity_keys = {(item.trip_id, item.frame_id) for item in parity}
    calibration_keys = {
        (item.trip_id, item.frame_id) for item in calibration
    }
    overlap = parity_keys & calibration_keys
    if overlap:
        raise DeploymentManifestError(
            f"parity and calibration selections overlap: {sorted(overlap)[:5]}"
        )
    if len(calibration_keys) != len(calibration):
        raise DeploymentManifestError("calibration selection contains duplicates")
    return parity, calibration


def _pair_manifest(
    *,
    kind: str,
    pairs: Sequence[StereoPair],
    trip_ids: Sequence[str],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    entries = [item.as_dict() for item in pairs]
    return {
        "schema": PAIR_MANIFEST_SCHEMA,
        "kind": kind,
        "openstereo_revision": OPENSTEREO_REVISION,
        "input_contract": {
            "native_shape_hwc": [NATIVE_HEIGHT, NATIVE_WIDTH, 3],
            "padded_shape_nchw": list(LIGHTSTEREO_INPUT_SHAPE),
            "input_names": list(OPENSTEREO_ONNX_INPUT_NAMES),
            "output_name": OPENSTEREO_ONNX_OUTPUT_NAME,
            "padding": "right-top edge replication",
            "normalization": "RGB ImageNet mean/std after division by 255",
        },
        "trip_ids": list(trip_ids),
        "selection": dict(selection),
        "entry_count": len(entries),
        "entries_sha256": canonical_sha256(entries),
        "entries": entries,
    }


def stereo_pair_content_sha256(
    data_root: Path, pairs: Sequence[StereoPair]
) -> str:
    """Hash selected paths and image bytes in stable manifest order."""
    root = data_root.expanduser().resolve()
    digest = hashlib.sha256()
    for pair in pairs:
        digest.update(_canonical_json_bytes(pair.as_dict()))
        for relative in (pair.left, pair.right):
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise DeploymentManifestError(
                    f"cannot hash stereo-pair image under {root}: {relative}"
                )
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def build_pair_manifests(
    pairs_by_trip: Mapping[str, Sequence[StereoPair]],
    *,
    trip_ids: Sequence[str] = TRIPS,
    parity_stride: int = PARITY_STRIDE,
    calibration_per_trip: int = CALIBRATION_PER_TRIP,
    require_phase02b_counts: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build deterministic manifest documents from already-discovered pairs."""
    parity, calibration = select_deployment_pairs(
        pairs_by_trip,
        trip_ids=trip_ids,
        parity_stride=parity_stride,
        calibration_per_trip=calibration_per_trip,
    )
    if require_phase02b_counts:
        if len(parity) != PARITY_TOTAL:
            raise DeploymentManifestError(
                f"parity selection must contain {PARITY_TOTAL} pairs; "
                f"found {len(parity)}"
            )
        if len(calibration) != CALIBRATION_TOTAL:
            raise DeploymentManifestError(
                f"calibration selection must contain {CALIBRATION_TOTAL} pairs; "
                f"found {len(calibration)}"
            )
        counts = Counter(item.trip_id for item in calibration)
        expected = {trip_id: CALIBRATION_PER_TRIP for trip_id in trip_ids}
        if dict(counts) != expected:
            raise DeploymentManifestError(
                f"calibration must contain 50 pairs per trip; found {dict(counts)}"
            )

    parity_manifest = _pair_manifest(
        kind=PARITY_KIND,
        pairs=parity,
        trip_ids=trip_ids,
        selection={
            "algorithm": "every_nth_ordered_frame",
            "stride": parity_stride,
            "expected_total": PARITY_TOTAL if require_phase02b_counts else len(parity),
        },
    )
    calibration_manifest = _pair_manifest(
        kind=CALIBRATION_KIND,
        pairs=calibration,
        trip_ids=trip_ids,
        selection={
            "algorithm": (
                "midpoint_of_equal_temporal_bins_after_exclusion_"
                "interleaved_by_trip"
            ),
            "pairs_per_trip": calibration_per_trip,
            "excluded_kind": PARITY_KIND,
            "excluded_entries_sha256": parity_manifest["entries_sha256"],
            "excluded_entries": parity_manifest["entries"],
            "expected_total": (
                CALIBRATION_TOTAL
                if require_phase02b_counts
                else len(calibration)
            ),
        },
    )
    return parity_manifest, calibration_manifest


def generate_pair_manifests(
    data_root: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Generate the frozen 72-pair parity and 300-pair calibration manifests."""
    root = data_root.expanduser().resolve()
    pairs_by_trip = discover_stereo_pairs(root)
    parity, calibration = build_pair_manifests(pairs_by_trip)
    parity_pairs = [StereoPair.from_dict(item) for item in parity["entries"]]
    calibration_pairs = [
        StereoPair.from_dict(item) for item in calibration["entries"]
    ]
    parity["content_sha256"] = stereo_pair_content_sha256(root, parity_pairs)
    calibration["content_sha256"] = stereo_pair_content_sha256(
        root, calibration_pairs
    )
    parity["content_hash_contract"] = (
        "sha256(canonical entry followed by left bytes and right bytes, "
        "in manifest order)"
    )
    calibration["content_hash_contract"] = parity["content_hash_contract"]
    output = output_dir.expanduser().resolve()
    parity_path = output / "lightstereo_parity_72.json"
    calibration_path = output / "lightstereo_int8_calibration_300.json"
    _write_json(parity_path, parity)
    _write_json(calibration_path, calibration)
    return parity_path, calibration_path


def load_pair_manifest(
    path: Path,
    *,
    expected_kind: str | None = None,
    expected_count: int | None = None,
) -> tuple[dict[str, Any], list[StereoPair]]:
    """Parse a manifest and verify its selection-integrity checksum."""
    resolved = path.expanduser().resolve()
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DeploymentManifestError(
            f"cannot read pair manifest {resolved}: {error}"
        ) from error
    if document.get("schema") != PAIR_MANIFEST_SCHEMA:
        raise DeploymentManifestError(
            f"{resolved}: unsupported schema {document.get('schema')!r}"
        )
    if expected_kind is not None and document.get("kind") != expected_kind:
        raise DeploymentManifestError(
            f"{resolved}: expected kind {expected_kind!r}, found "
            f"{document.get('kind')!r}"
        )
    if document.get("openstereo_revision") != OPENSTEREO_REVISION:
        raise DeploymentManifestError(
            f"{resolved}: expected OpenStereo revision {OPENSTEREO_REVISION}"
        )
    input_contract = document.get("input_contract", {})
    if input_contract.get("padded_shape_nchw") != list(
        LIGHTSTEREO_INPUT_SHAPE
    ):
        raise DeploymentManifestError(
            f"{resolved}: unexpected padded input shape"
        )
    if input_contract.get("input_names") != list(
        OPENSTEREO_ONNX_INPUT_NAMES
    ):
        raise DeploymentManifestError(f"{resolved}: unexpected input names")
    if input_contract.get("output_name") != OPENSTEREO_ONNX_OUTPUT_NAME:
        raise DeploymentManifestError(f"{resolved}: unexpected output name")
    raw_entries = document.get("entries")
    if not isinstance(raw_entries, list):
        raise DeploymentManifestError(f"{resolved}: entries must be a list")
    if document.get("entry_count") != len(raw_entries):
        raise DeploymentManifestError(f"{resolved}: entry_count is inconsistent")
    if expected_count is not None and len(raw_entries) != expected_count:
        raise DeploymentManifestError(
            f"{resolved}: expected {expected_count} entries, found "
            f"{len(raw_entries)}"
        )
    actual_hash = canonical_sha256(raw_entries)
    if document.get("entries_sha256") != actual_hash:
        raise DeploymentManifestError(
            f"{resolved}: entries_sha256 does not match manifest contents"
        )
    pairs = [StereoPair.from_dict(item) for item in raw_entries]
    keys = {(item.trip_id, item.frame_id) for item in pairs}
    if len(keys) != len(pairs):
        raise DeploymentManifestError(f"{resolved}: duplicate stereo pair")
    if document.get("kind") == CALIBRATION_KIND:
        selection = document.get("selection", {})
        excluded_entries = selection.get("excluded_entries")
        if not isinstance(excluded_entries, list):
            raise DeploymentManifestError(
                f"{resolved}: calibration manifest lacks excluded parity entries"
            )
        if canonical_sha256(excluded_entries) != selection.get(
            "excluded_entries_sha256"
        ):
            raise DeploymentManifestError(
                f"{resolved}: excluded parity checksum is inconsistent"
            )
        excluded_pairs = [
            StereoPair.from_dict(item) for item in excluded_entries
        ]
        if len(excluded_pairs) != PARITY_TOTAL:
            raise DeploymentManifestError(
                f"{resolved}: expected {PARITY_TOTAL} excluded parity pairs"
            )
        excluded_keys = {
            (item.trip_id, item.frame_id) for item in excluded_pairs
        }
        overlap = keys & excluded_keys
        if overlap:
            raise DeploymentManifestError(
                f"{resolved}: calibration overlaps parity: "
                f"{sorted(overlap)[:5]}"
            )
    return document, pairs


def resolve_manifest_pairs(
    manifest_path: Path,
    data_root: Path,
    *,
    expected_kind: str,
    expected_count: int,
) -> tuple[dict[str, Any], list[tuple[Path, Path]]]:
    """Resolve safe dataset-relative manifest paths and require every image."""
    document, pairs = load_pair_manifest(
        manifest_path,
        expected_kind=expected_kind,
        expected_count=expected_count,
    )
    root = data_root.expanduser().resolve()
    resolved_pairs = []
    for pair in pairs:
        left = (root / pair.left).resolve()
        right = (root / pair.right).resolve()
        if not left.is_relative_to(root) or not right.is_relative_to(root):
            raise DeploymentManifestError(
                f"pair {pair.trip_id}/{pair.frame_id} escapes data root"
            )
        if not left.is_file() or not right.is_file():
            raise DeploymentManifestError(
                f"pair {pair.trip_id}/{pair.frame_id} image missing: "
                f"{left} or {right}"
            )
        resolved_pairs.append((left, right))
    content_sha256 = document.get("content_sha256")
    if not isinstance(content_sha256, str) or len(content_sha256) != 64:
        raise DeploymentManifestError(
            f"{manifest_path}: missing selected-image content_sha256"
        )
    actual_content_sha256 = stereo_pair_content_sha256(root, pairs)
    if actual_content_sha256 != content_sha256:
        raise DeploymentManifestError(
            f"{manifest_path}: selected image content hash changed"
        )
    return document, resolved_pairs


def _optional_import(module_name: str, install_hint: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except (ImportError, OSError) as error:
        raise OptionalDependencyError(
            f"{module_name!r} is required for this deployment command. "
            f"{install_hint}. Original error: {error}"
        ) from error


def openstereo_export_command(
    *,
    openstereo_root: Path,
    checkpoint_path: Path,
    device: str = "0",
    simplify: bool = False,
) -> list[str]:
    """Return the audited LightStereo-only wrapper command."""
    root = openstereo_root.expanduser().resolve()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "export-onnx",
        "--openstereo-root",
        str(root),
        "--checkpoint",
        str(checkpoint_path.expanduser().resolve()),
        "--output",
        str(checkpoint_path.expanduser().resolve().with_suffix(".onnx")),
        "--device",
        str(device),
    ]
    if simplify:
        command.append("--simplify")
    return command


def validate_onnx_artifact(path: Path, onnx_module: Any | None = None) -> Any:
    """Run ONNX's checker and enforce the fixed LightStereo deployment I/O."""
    model_path = path.expanduser().resolve()
    if not model_path.is_file():
        raise BackendConfigurationError(f"ONNX model not found: {model_path}")
    onnx = onnx_module or _optional_import(
        "onnx", "install the pinned OpenStereo ONNX requirements"
    )
    try:
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)
        validate_lightstereo_onnx_model(model)
        output_names = {item.name for item in model.graph.output}
        if output_names != {OPENSTEREO_ONNX_OUTPUT_NAME}:
            raise BackendConfigurationError(
                "LightStereo ONNX output must be exactly "
                f"{OPENSTEREO_ONNX_OUTPUT_NAME!r}; found "
                f"{sorted(output_names)}"
            )
        expected_element_type = int(onnx.TensorProto.FLOAT)
        tensors = [*model.graph.input, *model.graph.output]
        wrong_types = {
            item.name: int(item.type.tensor_type.elem_type)
            for item in tensors
            if int(item.type.tensor_type.elem_type) != expected_element_type
        }
        if wrong_types:
            raise BackendConfigurationError(
                f"LightStereo ONNX tensors must be float32; found {wrong_types}"
            )
    except BackendConfigurationError:
        raise
    except Exception as error:
        raise BackendConfigurationError(
            f"ONNX validation failed for {model_path}: {error}"
        ) from error
    return model


def _dependency_versions(modules: Mapping[str, Any]) -> dict[str, str]:
    return {
        name: str(getattr(module, "__version__", "unknown"))
        for name, module in modules.items()
    }


def write_artifact_manifest(
    *,
    artifact_path: Path,
    artifact_kind: str,
    generation_command: Sequence[str],
    metadata: Mapping[str, Any],
    manifest_path: Path | None = None,
) -> Path:
    """Write a SHA-256 sidecar describing how a generated artifact was built."""
    artifact = artifact_path.expanduser().resolve()
    if not artifact.is_file():
        raise BackendConfigurationError(
            f"cannot manifest missing artifact: {artifact}"
        )
    destination = (
        manifest_path.expanduser().resolve()
        if manifest_path is not None
        else artifact.with_name(f"{artifact.name}.manifest.json")
    )
    document = {
        "schema": ARTIFACT_MANIFEST_SCHEMA,
        "artifact": {
            "name": artifact.name,
            "kind": artifact_kind,
            "bytes": artifact.stat().st_size,
            "sha256": sha256_file(artifact),
        },
        "generation_command_argv": [str(item) for item in generation_command],
        "metadata": dict(metadata),
    }
    _write_json(destination, document)
    return destination


def load_onnx_build_provenance(onnx_path: Path) -> dict[str, Any]:
    """Verify the exporter sidecar before using ONNX as an engine source."""
    onnx_file = onnx_path.expanduser().resolve()
    sidecar = onnx_file.with_name(f"{onnx_file.name}.manifest.json")
    try:
        document = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BackendConfigurationError(
            f"verified ONNX provenance sidecar is required: {sidecar}"
        ) from error
    metadata = document.get("metadata", {})
    checkpoint = metadata.get("checkpoint", {})
    config = metadata.get("config", {})
    if (
        document.get("schema") != ARTIFACT_MANIFEST_SCHEMA
        or document.get("artifact", {}).get("sha256") != sha256_file(onnx_file)
        or metadata.get("backend") != "lightstereo-onnx"
        or metadata.get("precision") != "fp32"
        or metadata.get("openstereo_revision") != OPENSTEREO_REVISION
        or checkpoint.get("sha256") != LIGHTSTEREO_CHECKPOINT_SHA256
        or config.get("relative_path")
        != LIGHTSTEREO_CONFIG_RELATIVE.as_posix()
        or not config.get("sha256")
    ):
        raise BackendConfigurationError(
            f"ONNX provenance sidecar is incomplete or inconsistent: {sidecar}"
        )
    return {
        "manifest_name": sidecar.name,
        "manifest_sha256": sha256_file(sidecar),
        "checkpoint": checkpoint,
        "config": config,
        "openstereo_revision": OPENSTEREO_REVISION,
    }


def export_lightstereo_onnx(
    *,
    openstereo_root: Path,
    checkpoint_path: Path,
    output_path: Path,
    device: str = "0",
    simplify: bool = False,
    generation_command: Sequence[str] | None = None,
) -> tuple[Path, Path]:
    """Run the pinned official exporter, validate it, and write a hash manifest."""
    root = openstereo_root.expanduser().resolve()
    checkpoint = checkpoint_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    verify_openstereo_revision(root)
    if not checkpoint.is_file():
        raise BackendConfigurationError(
            f"LightStereo-S checkpoint not found: {checkpoint}"
        )
    checkpoint_sha256 = verify_lightstereo_checkpoint(checkpoint)
    pinned_config = resolve_pinned_lightstereo_config(root)
    if simplify:
        raise BackendConfigurationError(
            "--simplify is disabled for the audited LightStereo-only export route"
        )
    torch = _optional_import(
        "torch", "activate the dedicated OpenStereo PyTorch environment"
    )
    yaml = _optional_import("yaml", "install the pinned PyYAML dependency")
    timm = _optional_import("timm", "install the pinned timm dependency")
    onnx = _optional_import(
        "onnx", "install the pinned OpenStereo ONNX requirements"
    )
    command = openstereo_export_command(
        openstereo_root=root,
        checkpoint_path=checkpoint,
        device=device,
        simplify=simplify,
    )
    if device == "cpu":
        torch_device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise BackendConfigurationError(
                "ONNX export requested CUDA but PyTorch cannot see it"
            )
        torch_device = torch.device(f"cuda:{int(device)}")
    model, source_provenance = instantiate_pinned_lightstereo(
        torch_module=torch,
        yaml_module=yaml,
        timm_module=timm,
        checkpoint_path=checkpoint,
        openstereo_root=root,
        config_path=pinned_config,
        device=torch_device,
    )

    class _OnnxWrapper(torch.nn.Module):
        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, left_img: Any, right_img: Any) -> Any:
            return self.wrapped(
                {"left": left_img, "right": right_img}
            )[OPENSTEREO_ONNX_OUTPUT_NAME]

    wrapper = _OnnxWrapper(model).eval()
    left = torch.zeros(LIGHTSTEREO_INPUT_SHAPE, device=torch_device)
    right = torch.zeros(LIGHTSTEREO_INPUT_SHAPE, device=torch_device)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp.onnx")
    try:
        torch.onnx.export(
            wrapper,
            (left, right),
            temporary,
            opset_version=17,
            do_constant_folding=True,
            input_names=list(OPENSTEREO_ONNX_INPUT_NAMES),
            output_names=[OPENSTEREO_ONNX_OUTPUT_NAME],
            dynamic_axes=None,
            dynamo=False,
        )
    except Exception as error:
        raise BackendConfigurationError(
            f"safe LightStereo-only ONNX export failed: {error}"
        ) from error
    if not temporary.is_file():
        raise BackendConfigurationError(
            "PyTorch reported success but did not create the ONNX artifact"
        )
    os.replace(temporary, output)
    validate_onnx_artifact(output, onnx_module=onnx)
    manifest = write_artifact_manifest(
        artifact_path=output,
        artifact_kind="lightstereo-s-onnx-opset17-static",
        generation_command=generation_command or command,
        metadata={
            "openstereo_revision": OPENSTEREO_REVISION,
            "openstereo_tracked_tree_clean": True,
            "backend": "lightstereo-onnx",
            "precision": "fp32",
            "checkpoint": {
                "name": checkpoint.name,
                "sha256": checkpoint_sha256,
                "expected_sha256": LIGHTSTEREO_CHECKPOINT_SHA256,
            },
            "config": {
                "relative_path": LIGHTSTEREO_CONFIG_RELATIVE.as_posix(),
                "sha256": sha256_file(pinned_config),
            },
            "source_provenance": source_provenance,
            "model_import_route": "guardian-lightstereo-only",
            "opset": 17,
            "input_names": list(OPENSTEREO_ONNX_INPUT_NAMES),
            "output_name": OPENSTEREO_ONNX_OUTPUT_NAME,
            "input_shape_nchw": list(LIGHTSTEREO_INPUT_SHAPE),
            "dynamic_axes": False,
            "dependencies": _dependency_versions(
                {"torch": torch, "onnx": onnx}
            ),
        },
    )
    return output, manifest


def prepare_lightstereo_image(image_bgr: np.ndarray) -> np.ndarray:
    """Apply pinned OpenStereo RGB normalization and 24-pixel top padding."""
    image = np.asarray(image_bgr)
    expected = (NATIVE_HEIGHT, NATIVE_WIDTH, 3)
    if image.shape != expected:
        raise DeploymentManifestError(
            f"calibration image must have native shape {expected}, got "
            f"{image.shape}; resizing calibration data is forbidden"
        )
    rgb = np.ascontiguousarray(image[..., ::-1], dtype=np.float32)
    padded = np.pad(
        rgb,
        ((INPUT_HEIGHT - NATIVE_HEIGHT, 0), (0, INPUT_WIDTH - NATIVE_WIDTH), (0, 0)),
        mode="edge",
    )
    normalized = (padded / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(
        normalized.transpose(2, 0, 1)[None], dtype=np.float32
    )


def _load_calibration_pair(
    left_path: Path, right_path: Path, cv2_module: Any
) -> tuple[np.ndarray, np.ndarray]:
    left = cv2_module.imread(str(left_path), cv2_module.IMREAD_COLOR)
    right = cv2_module.imread(str(right_path), cv2_module.IMREAD_COLOR)
    if left is None or right is None:
        raise DeploymentManifestError(
            f"OpenCV could not decode calibration pair {left_path}, {right_path}"
        )
    return prepare_lightstereo_image(left), prepare_lightstereo_image(right)


def _cache_metadata_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.name}.manifest.json")


def _cache_matches(cache_path: Path, expected_metadata: Mapping[str, Any]) -> bool:
    metadata_path = _cache_metadata_path(cache_path)
    if not cache_path.is_file() or cache_path.stat().st_size == 0:
        return False
    try:
        actual = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return actual == dict(expected_metadata)


def _make_entropy_calibrator(
    *,
    trt: Any,
    torch: Any,
    cv2: Any,
    pairs: Sequence[tuple[Path, Path]],
    cache_path: Path,
    cache_metadata: Mapping[str, Any],
    device_id: int,
) -> Any:
    """Create a TensorRT entropy calibrator backed by persistent CUDA tensors."""

    class LightStereoEntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self) -> None:
            super().__init__()
            self._index = 0
            self._device = torch.device(f"cuda:{int(device_id)}")
            self._buffers = {
                name: torch.empty(
                    LIGHTSTEREO_INPUT_SHAPE,
                    dtype=torch.float32,
                    device=self._device,
                )
                for name in OPENSTEREO_ONNX_INPUT_NAMES
            }
            self.cache_reused = False
            self.cache_written = False

        def get_batch_size(self) -> int:
            return 1

        def get_batch(self, names: Sequence[str]) -> list[int] | None:
            if self._index >= len(pairs):
                return None
            if set(names) != set(OPENSTEREO_ONNX_INPUT_NAMES):
                raise DeploymentManifestError(
                    "TensorRT requested unexpected calibration inputs "
                    f"{list(names)}"
                )
            left, right = _load_calibration_pair(
                *pairs[self._index], cv2_module=cv2
            )
            host_by_name = {
                OPENSTEREO_ONNX_INPUT_NAMES[0]: left,
                OPENSTEREO_ONNX_INPUT_NAMES[1]: right,
            }
            for name in names:
                host_tensor = torch.from_numpy(host_by_name[name])
                self._buffers[name].copy_(host_tensor)
            torch.cuda.synchronize(self._device)
            self._index += 1
            return [int(self._buffers[name].data_ptr()) for name in names]

        def read_calibration_cache(self) -> bytes | None:
            if _cache_matches(cache_path, cache_metadata):
                self.cache_reused = True
                return cache_path.read_bytes()
            return None

        def write_calibration_cache(self, cache: bytes) -> None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_name(f".{cache_path.name}.tmp")
            temporary.write_bytes(bytes(cache))
            os.replace(temporary, cache_path)
            _write_json(_cache_metadata_path(cache_path), dict(cache_metadata))
            self.cache_written = True

    return LightStereoEntropyCalibrator()


def validate_tensorrt_network_contract(network: Any) -> None:
    """Require the same static names and shapes after TensorRT ONNX parsing."""
    inputs = {
        network.get_input(index).name: tuple(
            int(item) for item in network.get_input(index).shape
        )
        for index in range(int(network.num_inputs))
    }
    expected_inputs = {
        name: LIGHTSTEREO_INPUT_SHAPE for name in OPENSTEREO_ONNX_INPUT_NAMES
    }
    if inputs != expected_inputs:
        raise BackendConfigurationError(
            f"TensorRT network inputs must be {expected_inputs}; found {inputs}"
        )
    outputs = {
        network.get_output(index).name: tuple(
            int(item) for item in network.get_output(index).shape
        )
        for index in range(int(network.num_outputs))
    }
    allowed_output_shapes = {
        (1, INPUT_HEIGHT, INPUT_WIDTH),
        (1, 1, INPUT_HEIGHT, INPUT_WIDTH),
    }
    if (
        set(outputs) != {OPENSTEREO_ONNX_OUTPUT_NAME}
        or outputs.get(OPENSTEREO_ONNX_OUTPUT_NAME)
        not in allowed_output_shapes
    ):
        raise BackendConfigurationError(
            "TensorRT network output must be exactly static disp_pred at "
            f"384x640; found {outputs}"
        )


def _parse_tensorrt_major(version: Any) -> int:
    try:
        return int(str(version).split(".", 1)[0])
    except (TypeError, ValueError) as error:
        raise OptionalDependencyError(
            f"cannot parse TensorRT version {version!r}"
        ) from error


def build_tensorrt_engine(
    *,
    onnx_path: Path,
    output_path: Path,
    precision: str,
    calibration_manifest_path: Path | None = None,
    data_root: Path | None = None,
    calibration_cache_path: Path | None = None,
    device_id: int = 0,
    generation_command: Sequence[str] | None = None,
) -> tuple[Path, Path]:
    """Build a TensorRT 10 FP16 or entropy-calibrated INT8 static engine."""
    if precision not in {"fp16", "int8"}:
        raise BackendConfigurationError("precision must be fp16 or int8")
    onnx_file = onnx_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    onnx = _optional_import("onnx", "install the deployment ONNX requirements")
    trt = _optional_import(
        "tensorrt", "install TensorRT 10.x matching CUDA inside WSL"
    )
    if _parse_tensorrt_major(getattr(trt, "__version__", None)) != 10:
        raise OptionalDependencyError(
            f"TensorRT 10.x is required; found "
            f"{getattr(trt, '__version__', 'unknown')}"
        )
    validate_onnx_artifact(onnx_file, onnx_module=onnx)
    onnx_provenance = load_onnx_build_provenance(onnx_file)

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_file)):
        errors = [
            str(parser.get_error(index))
            for index in range(int(parser.num_errors))
        ]
        raise BackendConfigurationError(
            "TensorRT failed to parse ONNX: " + " | ".join(errors)
        )
    validate_tensorrt_network_contract(network)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, TENSORRT_WORKSPACE_BYTES
    )
    calibrator = None
    calibration_metadata: dict[str, Any] | None = None
    dependency_modules: dict[str, Any] = {"onnx": onnx, "tensorrt": trt}

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise BackendConfigurationError(
                "this GPU does not report fast native FP16 support"
            )
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        if calibration_manifest_path is None or data_root is None:
            raise BackendConfigurationError(
                "INT8 requires --calibration-manifest and --data-root"
            )
        if calibration_cache_path is None:
            calibration_cache_path = output.with_suffix(
                ".calibration.cache"
            )
        cache = calibration_cache_path.expanduser().resolve()
        manifest_path = calibration_manifest_path.expanduser().resolve()
        manifest, resolved_pairs = resolve_manifest_pairs(
            manifest_path,
            data_root,
            expected_kind=CALIBRATION_KIND,
            expected_count=CALIBRATION_TOTAL,
        )
        per_trip = Counter(
            str(item["trip_id"]) for item in manifest["entries"]
        )
        expected_per_trip = {
            trip_id: CALIBRATION_PER_TRIP for trip_id in TRIPS
        }
        if dict(per_trip) != expected_per_trip:
            raise DeploymentManifestError(
                f"INT8 manifest must contain 50 pairs per trip; found "
                f"{dict(per_trip)}"
            )
        torch = _optional_import(
            "torch", "install CUDA-enabled PyTorch for calibration buffers"
        )
        cv2 = _optional_import(
            "cv2", "install OpenCV in the OpenStereo environment"
        )
        if not torch.cuda.is_available():
            raise BackendConfigurationError(
                "INT8 calibration requires CUDA-visible PyTorch"
            )
        torch.cuda.set_device(int(device_id))
        if not builder.platform_has_fast_int8:
            raise BackendConfigurationError(
                "this GPU does not report fast native INT8 support"
            )
        cache_metadata = {
            "schema": "guardian.phase02b.tensorrt-calibration-cache.v1",
            "tensorrt_version": str(trt.__version__),
            "calibration_algorithm": "IInt8EntropyCalibrator2",
            "onnx_sha256": sha256_file(onnx_file),
            "calibration_manifest_sha256": sha256_file(manifest_path),
            "calibration_entries_sha256": manifest["entries_sha256"],
            "calibration_content_sha256": manifest["content_sha256"],
            "batch_size": 1,
            "batch_count": len(resolved_pairs),
            "input_shape_nchw": list(LIGHTSTEREO_INPUT_SHAPE),
            "input_names": list(OPENSTEREO_ONNX_INPUT_NAMES),
            "preprocessing": (
                "BGR-to-RGB; right-top edge pad to 384x640; "
                "ImageNet normalization"
            ),
        }
        calibrator = _make_entropy_calibrator(
            trt=trt,
            torch=torch,
            cv2=cv2,
            pairs=resolved_pairs,
            cache_path=cache,
            cache_metadata=cache_metadata,
            device_id=device_id,
        )
        config.set_flag(trt.BuilderFlag.INT8)
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        config.int8_calibrator = calibrator
        calibration_metadata = {
            "manifest": {
                "name": manifest_path.name,
                "sha256": sha256_file(manifest_path),
                "entries_sha256": manifest["entries_sha256"],
                "content_sha256": manifest["content_sha256"],
                "pair_count": len(resolved_pairs),
                "pairs_per_trip": expected_per_trip,
            },
            "cache": {
                "name": cache.name,
                "expected_metadata": cache_metadata,
            },
            "algorithm": "IInt8EntropyCalibrator2",
            "fp16_fallback_enabled": bool(builder.platform_has_fast_fp16),
        }
        dependency_modules.update({"torch": torch, "opencv": cv2})

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise BackendConfigurationError(
            f"TensorRT returned no serialized {precision} engine"
        )

    if calibration_metadata is not None and calibrator is not None:
        calibration_metadata["cache"].update(
            {
                "reused": bool(calibrator.cache_reused),
                "written": bool(calibrator.cache_written),
            }
        )
        cache_name = calibration_metadata["cache"]["name"]
        cache_file = (
            calibration_cache_path.expanduser().resolve()
            if calibration_cache_path is not None
            else output.with_suffix(".calibration.cache")
        )
        if not cache_file.is_file() or cache_file.stat().st_size == 0:
            raise BackendConfigurationError(
                f"TensorRT did not produce a non-empty calibration cache "
                f"{cache_name}; refusing to label the engine INT8 calibrated"
            )
        calibration_metadata["cache"]["sha256"] = sha256_file(cache_file)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_bytes(bytes(serialized))
    os.replace(temporary, output)

    default_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "build-engine",
        "--onnx",
        str(onnx_file),
        "--output",
        str(output),
        "--precision",
        precision,
        "--device-id",
        str(device_id),
    ]
    if precision == "int8":
        assert calibration_manifest_path is not None
        assert data_root is not None
        assert calibration_cache_path is not None
        default_command.extend(
            [
                "--calibration-manifest",
                str(calibration_manifest_path.expanduser().resolve()),
                "--calibration-cache",
                str(calibration_cache_path.expanduser().resolve()),
                "--data-root",
                str(data_root.expanduser().resolve()),
            ]
        )
    manifest_path = write_artifact_manifest(
        artifact_path=output,
        artifact_kind=f"lightstereo-s-tensorrt10-{precision}",
        generation_command=generation_command or default_command,
        metadata={
            "backend": "lightstereo-tensorrt",
            "openstereo_revision": OPENSTEREO_REVISION,
            "checkpoint": onnx_provenance["checkpoint"],
            "config": onnx_provenance["config"],
            "source_onnx": {
                "name": onnx_file.name,
                "sha256": sha256_file(onnx_file),
                "manifest_name": onnx_provenance["manifest_name"],
                "manifest_sha256": onnx_provenance["manifest_sha256"],
            },
            "input_shape_nchw": list(LIGHTSTEREO_INPUT_SHAPE),
            "input_names": list(OPENSTEREO_ONNX_INPUT_NAMES),
            "output_name": OPENSTEREO_ONNX_OUTPUT_NAME,
            "precision": precision,
            "workspace_bytes": TENSORRT_WORKSPACE_BYTES,
            "calibration": calibration_metadata,
            "dependencies": _dependency_versions(dependency_modules),
        },
    )
    return output, manifest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Phase 2B LightStereo manifests, static ONNX, and "
            "TensorRT 10 engines."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifests = subparsers.add_parser(
        "generate-manifests",
        help="freeze the 72-pair parity and 300-pair INT8 selections",
    )
    manifests.add_argument("--data-root", type=Path, required=True)
    manifests.add_argument("--output-dir", type=Path, required=True)

    validate = subparsers.add_parser(
        "validate-onnx", help="check static opset-17 names and shapes"
    )
    validate.add_argument("--onnx", type=Path, required=True)

    export = subparsers.add_parser(
        "export-onnx", help="run the pinned OpenStereo LightStereo-S exporter"
    )
    export.add_argument("--openstereo-root", type=Path, required=True)
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--device", default="0")
    export.add_argument("--simplify", action="store_true")

    engine = subparsers.add_parser(
        "build-engine", help="build a fixed TensorRT 10 FP16 or calibrated INT8 engine"
    )
    engine.add_argument("--onnx", type=Path, required=True)
    engine.add_argument("--output", type=Path, required=True)
    engine.add_argument("--precision", choices=("fp16", "int8"), required=True)
    engine.add_argument("--calibration-manifest", type=Path)
    engine.add_argument("--data-root", type=Path)
    engine.add_argument("--calibration-cache", type=Path)
    engine.add_argument("--device-id", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    command_argv = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    try:
        if args.command == "generate-manifests":
            parity, calibration = generate_pair_manifests(
                args.data_root, args.output_dir
            )
            print(
                json.dumps(
                    {
                        "parity_manifest": str(parity),
                        "parity_sha256": sha256_file(parity),
                        "calibration_manifest": str(calibration),
                        "calibration_sha256": sha256_file(calibration),
                    },
                    indent=2,
                )
            )
        elif args.command == "validate-onnx":
            validate_onnx_artifact(args.onnx)
            print(
                json.dumps(
                    {
                        "onnx": str(args.onnx.expanduser().resolve()),
                        "sha256": sha256_file(args.onnx.expanduser().resolve()),
                        "valid": True,
                    },
                    indent=2,
                )
            )
        elif args.command == "export-onnx":
            artifact, manifest = export_lightstereo_onnx(
                openstereo_root=args.openstereo_root,
                checkpoint_path=args.checkpoint,
                output_path=args.output,
                device=args.device,
                simplify=args.simplify,
                generation_command=command_argv,
            )
            print(json.dumps({"onnx": str(artifact), "manifest": str(manifest)}, indent=2))
        elif args.command == "build-engine":
            artifact, manifest = build_tensorrt_engine(
                onnx_path=args.onnx,
                output_path=args.output,
                precision=args.precision,
                calibration_manifest_path=args.calibration_manifest,
                data_root=args.data_root,
                calibration_cache_path=args.calibration_cache,
                device_id=args.device_id,
                generation_command=command_argv,
            )
            print(
                json.dumps(
                    {"engine": str(artifact), "manifest": str(manifest)},
                    indent=2,
                )
            )
        else:
            parser.error(f"unhandled command {args.command}")
    except (BackendConfigurationError, DeploymentManifestError) as error:
        parser.exit(2, f"ERROR: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
