"""Exercise the exact Guardian -> pinned OpenStereo LightStereo-S path.

This is an integration smoke, not a benchmark. It loads the reviewed config and
checkpoint through Guardian's PyTorch backend, runs one native-size pair on
CUDA, and validates the backend-neutral output contract.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


OPENSTEREO_REVISION = "23d71c92e33ad1f80dfc42bf29f5c6a914d38769"
LIGHTSTEREO_CONFIG_RELATIVE = Path(
    "cfgs/lightstereo/lightstereo_s_kitti.yaml"
)
EXPECTED_MODEL_CONFIG: dict[str, Any] = {
    "NAME": "LightStereo",
    "MAX_DISP": 192,
    "EXPANSE_RATIO": 4,
    "AGGREGATION_BLOCKS": [1, 2, 4],
    "LEFT_ATT": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construct and run the pinned LightStereo-S CUDA backend."
    )
    parser.add_argument(
        "--openstereo-root",
        type=Path,
        required=True,
        help="Pinned external OpenStereo checkout.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Verified external LightStereo-S KITTI checkpoint.",
    )
    parser.add_argument(
        "--guardian-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Guardian repository root.",
    )
    return parser.parse_args()


def require_revision(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip().lower()
    if revision != OPENSTEREO_REVISION:
        raise RuntimeError(
            "OpenStereo revision mismatch: expected "
            f"{OPENSTEREO_REVISION}, found {revision}. Run "
            f"`git -C {root} switch --detach {OPENSTEREO_REVISION}`."
        )
    return revision


def require_lightstereo_config(root: Path) -> Path:
    config_path = root / LIGHTSTEREO_CONFIG_RELATIVE
    if not config_path.is_file():
        raise RuntimeError(
            f"Pinned LightStereo-S KITTI config is missing: {config_path}"
        )
    with config_path.open("r", encoding="utf-8") as stream:
        configuration = yaml.safe_load(stream)
    model_config = configuration.get("MODEL", {})
    mismatches = {
        key: {"expected": expected, "actual": model_config.get(key)}
        for key, expected in EXPECTED_MODEL_CONFIG.items()
        if model_config.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            "Pinned LightStereo-S model config changed: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return config_path


def import_guardian_backend(guardian_root: Path) -> Any:
    source_root = (
        guardian_root
        / "ai_cv"
        / "phases"
        / "02_detection_tracking"
        / "src"
    )
    backend_path = source_root / "stereo_backends.py"
    if not backend_path.is_file():
        raise RuntimeError(f"Guardian stereo adapter is missing: {backend_path}")
    sys.path.insert(0, str(source_root))
    try:
        from stereo_backends import (  # type: ignore[import-not-found]
            LIGHTSTEREO_INPUT_SHAPE,
            LightStereoPyTorchBackend,
        )
    except Exception as error:
        raise RuntimeError(
            "Guardian's LightStereo adapter could not be imported. Run this "
            "smoke from the dedicated OpenStereo environment and keep Guardian "
            f"at the Phase 2B revision. Original error: {error}"
        ) from error
    return LIGHTSTEREO_INPUT_SHAPE, LightStereoPyTorchBackend


def construction_failure(error: Exception) -> RuntimeError:
    message = str(error)
    optional_aggregator_markers = (
        "FoundationStereo",
        "FastFoundationStereo",
        "flash_attn",
        "timm>=0.9",
    )
    if any(marker in message for marker in optional_aggregator_markers):
        return RuntimeError(
            "Pinned OpenStereo's `stereo.modeling` package imported an "
            "unrelated optional model before LightStereo-S. Do not install "
            "FoundationStereo/flash-attn for this benchmark. The Guardian "
            "adapter must use its LightStereo-only import path. Original "
            f"error: {type(error).__name__}: {error}"
        )
    return RuntimeError(
        "LightStereo-S construction/checkpoint load failed. Verify the pinned "
        "checkout, dedicated requirements, checkpoint SHA-256, CUDA-enabled "
        "PyTorch, and access to the timm MobileNetV2 backbone cache. Original "
        f"error: {type(error).__name__}: {error}"
    )


def main() -> int:
    args = parse_args()
    openstereo_root = args.openstereo_root.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    guardian_root = args.guardian_root.expanduser().resolve()

    if not checkpoint_path.is_file():
        raise RuntimeError(
            f"LightStereo-S KITTI checkpoint is missing: {checkpoint_path}"
        )
    revision = require_revision(openstereo_root)
    config_path = require_lightstereo_config(openstereo_root)
    input_shape, backend_class = import_guardian_backend(guardian_root)

    backend = None
    try:
        try:
            backend = backend_class(
                checkpoint_path=checkpoint_path,
                openstereo_root=openstereo_root,
                config_path=config_path,
                precision="fp32",
                device="cuda:0",
            )
        except Exception as error:
            raise construction_failure(error) from error

        # The deployment contract is fixed at padded 1x3x384x640, so a smaller
        # tensor would not prove that the intended graph path is runnable.
        left = np.zeros((360, 640, 3), dtype=np.uint8)
        right = np.zeros_like(left)
        try:
            result = backend.infer(left, right)
        except Exception as error:
            raise RuntimeError(
                "LightStereo-S constructed and loaded its checkpoint, but the "
                "native 640x360 CUDA forward failed. Check CUDA memory/runtime "
                f"compatibility. Original error: {type(error).__name__}: {error}"
            ) from error

        if tuple(result.input_shape) != tuple(input_shape):
            raise RuntimeError(
                f"Unexpected model input shape: {result.input_shape} != {input_shape}"
            )
        if result.disparity_px.shape != (360, 640):
            raise RuntimeError(
                "LightStereo-S output did not restore to native 360x640: "
                f"{result.disparity_px.shape}"
            )
        if not np.all(np.isfinite(result.disparity_px)):
            raise RuntimeError("LightStereo-S output contains NaN or Inf values.")

        print(
            json.dumps(
                {
                    "backend": result.backend,
                    "checkpoint_sha256": result.model_sha256,
                    "config": str(config_path),
                    "input_shape": list(result.input_shape),
                    "openstereo_revision": revision,
                    "output_shape": list(result.disparity_px.shape),
                    "peak_gpu_memory_mb": backend.peak_gpu_memory_mb(),
                    "precision": result.precision,
                    "stereo_total_ms": result.timings_ms.get("stereo_total"),
                    "valid_fraction": float(np.mean(result.valid_mask)),
                },
                sort_keys=True,
            )
        )
    finally:
        if backend is not None:
            backend.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"LightStereo integration smoke failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
