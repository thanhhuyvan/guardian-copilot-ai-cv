"""Validate the immutable V1 runtime contract before containerisation."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    actual_hash = sha256(args.model_path)
    expected_hash = config["model"]["sha256"]
    if actual_hash != expected_hash:
        raise ValueError(f"model SHA-256 mismatch: expected {expected_hash}, got {actual_hash}")
    import cv2
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; V1 release requires the validated GPU path")
    try:
        from jsonschema import Draft202012Validator
    except ImportError as error:
        raise RuntimeError("jsonschema is required for contract preflight") from error

    manifest = {
        "schema_version": "run_manifest.v1",
        "run_id": "guardian-v1-preflight",
        "created_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "code_commit": git_commit(args.repository.resolve()),
        "model_version": f"{config['release']}:{actual_hash[:12]}",
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "dataset_id": "runtime-stereo-stream",
        "processing_mode": config["safety"]["processing_mode"],
        "input_features": ["stereo_left", "stereo_right", *config["input"]["ego_telemetry"]],
        "uses_future_frames": config["safety"]["uses_future_frames"],
        "uses_full_event_schedule": config["safety"]["uses_full_event_schedule"],
        "depth_keyframe_policy": config["safety"]["depth_keyframe_policy"],
        "hardware": {
            "platform": f"{platform.system()} {platform.release()}",
            "accelerator": torch.cuda.get_device_name(0),
        },
        "contract_versions": config["contracts"],
    }
    schema_path = args.repository / "ai_cv" / "shared" / "contracts" / "run_manifest.v1.schema.json"
    Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(manifest)
    document = {
        "passed": True,
        "checks": {
            "model_sha256": actual_hash,
            "cuda_available": True,
            "opencv_version": cv2.__version__,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "run_manifest": manifest,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
