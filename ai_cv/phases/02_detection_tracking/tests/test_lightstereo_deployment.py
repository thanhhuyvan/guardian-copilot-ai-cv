from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import lightstereo_deployment as DEPLOY  # noqa: E402


def _pairs(trip_id: str, count: int = 600) -> list[DEPLOY.StereoPair]:
    return [
        DEPLOY.StereoPair(
            trip_id=trip_id,
            frame_id=frame_id,
            left=f"{trip_id}/kitti/image_2/{frame_id:06d}.jpg",
            right=f"{trip_id}/kitti/image_3/{frame_id:06d}.jpg",
        )
        for frame_id in range(count)
    ]


class PairSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pairs_by_trip = {
            trip_id: _pairs(trip_id) for trip_id in DEPLOY.TRIPS
        }

    def test_phase02b_selection_is_exact_disjoint_and_deterministic(self) -> None:
        parity, calibration = DEPLOY.select_deployment_pairs(
            self.pairs_by_trip
        )
        repeated = DEPLOY.select_deployment_pairs(self.pairs_by_trip)

        self.assertEqual(len(parity), 72)
        self.assertEqual(len(calibration), 300)
        self.assertEqual((parity, calibration), repeated)
        self.assertEqual(
            [item.trip_id for item in calibration[:6]],
            list(DEPLOY.TRIPS),
        )
        parity_keys = {(item.trip_id, item.frame_id) for item in parity}
        calibration_keys = {
            (item.trip_id, item.frame_id) for item in calibration
        }
        self.assertFalse(parity_keys & calibration_keys)
        for trip_id in DEPLOY.TRIPS:
            self.assertEqual(
                [
                    item.frame_id
                    for item in parity
                    if item.trip_id == trip_id
                ],
                list(range(0, 600, 50)),
            )
            self.assertEqual(
                sum(item.trip_id == trip_id for item in calibration), 50
            )

    def test_manifests_have_stable_content_hashes_and_exclusion_link(self) -> None:
        parity, calibration = DEPLOY.build_pair_manifests(
            self.pairs_by_trip
        )
        repeated = DEPLOY.build_pair_manifests(self.pairs_by_trip)

        self.assertEqual((parity, calibration), repeated)
        self.assertEqual(parity["entry_count"], 72)
        self.assertEqual(calibration["entry_count"], 300)
        self.assertEqual(
            calibration["selection"]["excluded_entries_sha256"],
            parity["entries_sha256"],
        )
        self.assertEqual(
            calibration["selection"]["excluded_entries"],
            parity["entries"],
        )
        self.assertEqual(
            DEPLOY.canonical_sha256(parity["entries"]),
            parity["entries_sha256"],
        )

    def test_manifest_loader_detects_entry_tampering(self) -> None:
        parity, _ = DEPLOY.build_pair_manifests(self.pairs_by_trip)
        parity["entries"][0]["frame_id"] = 999
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "parity.json"
            path.write_text(json.dumps(parity), encoding="utf-8")

            with self.assertRaisesRegex(
                DEPLOY.DeploymentManifestError, "entries_sha256"
            ):
                DEPLOY.load_pair_manifest(path)

    def test_stereo_pair_rejects_path_traversal(self) -> None:
        with self.assertRaisesRegex(ValueError, "dataset-relative"):
            DEPLOY.StereoPair(
                trip_id="T01-Sample",
                frame_id=0,
                left="../secret.jpg",
                right="T01-Sample/right.jpg",
            )

    def test_calibration_loader_rejects_parity_overlap(self) -> None:
        parity, calibration = DEPLOY.build_pair_manifests(
            self.pairs_by_trip
        )
        calibration["entries"][0] = parity["entries"][0]
        calibration["entries_sha256"] = DEPLOY.canonical_sha256(
            calibration["entries"]
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calibration.json"
            path.write_text(json.dumps(calibration), encoding="utf-8")

            with self.assertRaisesRegex(
                DEPLOY.DeploymentManifestError, "overlaps parity"
            ):
                DEPLOY.load_pair_manifest(
                    path,
                    expected_kind=DEPLOY.CALIBRATION_KIND,
                    expected_count=300,
                )


class ExportContractTests(unittest.TestCase):
    def test_export_command_uses_audited_lightstereo_only_wrapper(self) -> None:
        command = DEPLOY.openstereo_export_command(
            openstereo_root=Path("/opt/OpenStereo"),
            checkpoint_path=Path("/models/lightstereo_s.pth"),
            device="0",
        )

        self.assertIn("lightstereo_deployment.py", command[1])
        self.assertEqual(command[2], "export-onnx")
        self.assertEqual(
            command[command.index("--openstereo-root") + 1],
            str(Path("/opt/OpenStereo").resolve()),
        )
        self.assertNotIn("--dynamic", command)
        self.assertNotIn("--half", command)
        self.assertNotIn("deploy/export.py", " ".join(command))

    def test_missing_optional_dependency_has_actionable_error(self) -> None:
        with mock.patch.object(
            DEPLOY.importlib,
            "import_module",
            side_effect=ImportError("not installed"),
        ):
            with self.assertRaisesRegex(
                DEPLOY.OptionalDependencyError, "install onnx"
            ):
                DEPLOY._optional_import("onnx", "install onnx")

    def test_artifact_manifest_records_hash_size_and_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "model.onnx"
            artifact.write_bytes(b"fake-onnx")

            sidecar = DEPLOY.write_artifact_manifest(
                artifact_path=artifact,
                artifact_kind="test",
                generation_command=["python", "export.py"],
                metadata={"opset": 17},
            )
            document = json.loads(sidecar.read_text(encoding="utf-8"))

            self.assertEqual(document["artifact"]["bytes"], 9)
            self.assertEqual(
                document["artifact"]["sha256"], DEPLOY.sha256_file(artifact)
            )
            self.assertEqual(
                document["generation_command_argv"],
                ["python", "export.py"],
            )

    def test_engine_source_requires_complete_onnx_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "model.onnx"
            artifact.write_bytes(b"verified-onnx")
            DEPLOY.write_artifact_manifest(
                artifact_path=artifact,
                artifact_kind="lightstereo-s-onnx-opset17-static",
                generation_command=["python", "export-onnx"],
                metadata={
                    "backend": "lightstereo-onnx",
                    "precision": "fp32",
                    "openstereo_revision": DEPLOY.OPENSTEREO_REVISION,
                    "checkpoint": {
                        "sha256": DEPLOY.LIGHTSTEREO_CHECKPOINT_SHA256
                    },
                    "config": {
                        "relative_path": (
                            DEPLOY.LIGHTSTEREO_CONFIG_RELATIVE.as_posix()
                        ),
                        "sha256": "a" * 64,
                    },
                },
            )

            provenance = DEPLOY.load_onnx_build_provenance(artifact)

            self.assertEqual(
                provenance["checkpoint"]["sha256"],
                DEPLOY.LIGHTSTEREO_CHECKPOINT_SHA256,
            )


class CalibrationTests(unittest.TestCase):
    def test_preprocessing_is_rgb_normalized_and_top_edge_padded(self) -> None:
        image = np.zeros((360, 640, 3), dtype=np.uint8)
        image[0, 0] = [10, 20, 30]

        output = DEPLOY.prepare_lightstereo_image(image)

        self.assertEqual(output.shape, (1, 3, 384, 640))
        expected_rgb = np.asarray([30, 20, 10], dtype=np.float32)
        expected = (
            expected_rgb / 255.0 - DEPLOY.IMAGENET_MEAN
        ) / DEPLOY.IMAGENET_STD
        np.testing.assert_allclose(output[0, :, 0, 0], expected, atol=1e-6)
        np.testing.assert_allclose(output[0, :, 24, 0], expected, atol=1e-6)

    def test_preprocessing_refuses_resizing(self) -> None:
        with self.assertRaisesRegex(
            DEPLOY.DeploymentManifestError, "resizing.*forbidden"
        ):
            DEPLOY.prepare_lightstereo_image(
                np.zeros((359, 640, 3), dtype=np.uint8)
            )

    def test_cache_reuse_requires_exact_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "lightstereo.calibration.cache"
            cache.write_bytes(b"calibration")
            metadata = {"onnx_sha256": "a" * 64, "pair_count": 300}
            DEPLOY._write_json(
                DEPLOY._cache_metadata_path(cache), metadata
            )

            self.assertTrue(DEPLOY._cache_matches(cache, metadata))
            self.assertFalse(
                DEPLOY._cache_matches(
                    cache, {"onnx_sha256": "b" * 64, "pair_count": 300}
                )
            )

    def test_selected_image_hash_changes_with_image_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = root / "T01-Sample/left.jpg"
            right = root / "T01-Sample/right.jpg"
            left.parent.mkdir(parents=True)
            left.write_bytes(b"left-v1")
            right.write_bytes(b"right-v1")
            pair = DEPLOY.StereoPair(
                trip_id="T01-Sample",
                frame_id=0,
                left="T01-Sample/left.jpg",
                right="T01-Sample/right.jpg",
            )
            original = DEPLOY.stereo_pair_content_sha256(root, [pair])

            right.write_bytes(b"right-v2")
            changed = DEPLOY.stereo_pair_content_sha256(root, [pair])

            self.assertNotEqual(original, changed)


class _Tensor:
    def __init__(self, name: str, shape: tuple[int, ...]) -> None:
        self.name = name
        self.shape = shape


class _Network:
    def __init__(
        self,
        inputs: list[_Tensor],
        outputs: list[_Tensor],
    ) -> None:
        self._inputs = inputs
        self._outputs = outputs
        self.num_inputs = len(inputs)
        self.num_outputs = len(outputs)

    def get_input(self, index: int) -> _Tensor:
        return self._inputs[index]

    def get_output(self, index: int) -> _Tensor:
        return self._outputs[index]


class TensorRtContractTests(unittest.TestCase):
    def test_network_contract_accepts_only_pinned_names_and_static_shape(self) -> None:
        network = _Network(
            [
                _Tensor("left_img", (1, 3, 384, 640)),
                _Tensor("right_img", (1, 3, 384, 640)),
            ],
            [_Tensor("disp_pred", (1, 384, 640))],
        )

        DEPLOY.validate_tensorrt_network_contract(network)

    def test_network_contract_rejects_dynamic_or_renamed_input(self) -> None:
        network = _Network(
            [
                _Tensor("left", (-1, 3, 384, 640)),
                _Tensor("right_img", (1, 3, 384, 640)),
            ],
            [_Tensor("disp_pred", (1, 384, 640))],
        )

        with self.assertRaisesRegex(
            DEPLOY.BackendConfigurationError, "network inputs"
        ):
            DEPLOY.validate_tensorrt_network_contract(network)

    def test_network_contract_rejects_dynamic_output(self) -> None:
        network = _Network(
            [
                _Tensor("left_img", (1, 3, 384, 640)),
                _Tensor("right_img", (1, 3, 384, 640)),
            ],
            [_Tensor("disp_pred", (1, -1, 640))],
        )

        with self.assertRaisesRegex(
            DEPLOY.BackendConfigurationError, "network output"
        ):
            DEPLOY.validate_tensorrt_network_contract(network)

    def test_requires_tensorrt_major_version_10(self) -> None:
        self.assertEqual(DEPLOY._parse_tensorrt_major("10.9.0"), 10)
        self.assertEqual(DEPLOY._parse_tensorrt_major("8.6.1"), 8)
        with self.assertRaises(DEPLOY.OptionalDependencyError):
            DEPLOY._parse_tensorrt_major(None)


if __name__ == "__main__":
    unittest.main()
