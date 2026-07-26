from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import stereo_backends as BACKENDS  # noqa: E402


def result(
    disparity: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    backend: str = "test",
) -> BACKENDS.StereoResult:
    if valid is None:
        valid = np.isfinite(disparity) & (disparity > 0)
    return BACKENDS.StereoResult(
        disparity_px=disparity,
        valid_mask=valid,
        confidence=None,
        backend=backend,
        precision="fp32",
        input_shape=(1, 3, disparity.shape[0], disparity.shape[1]),
        model_sha256=None,
        timings_ms={"stereo_total": 1.0},
    )


class StereoResultContractTests(unittest.TestCase):
    def test_coerces_public_arrays_to_contract_dtypes(self) -> None:
        value = BACKENDS.StereoResult(
            disparity_px=np.array([[2, -1]], dtype=np.float64),
            valid_mask=np.array([[1, 0]], dtype=np.uint8),
            confidence=np.array([[0.75, 0.0]], dtype=np.float64),
            backend="fake",
            precision="fp32",
            input_shape=(1, 3, 1, 2),
            model_sha256="a" * 64,
            timings_ms={"inference": 2},
        )

        self.assertEqual(value.disparity_px.dtype, np.float32)
        self.assertEqual(value.valid_mask.dtype, np.bool_)
        assert value.confidence is not None
        self.assertEqual(value.confidence.dtype, np.float32)
        self.assertTrue(value.disparity_px.flags.c_contiguous)

    def test_rejects_valid_nonpositive_or_nonfinite_disparity(self) -> None:
        for invalid in (0.0, -1.0, np.nan):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    result(
                        np.array([[invalid]], dtype=np.float32),
                        np.ones((1, 1), dtype=bool),
                    )

    def test_rejects_nonfinite_disparity_even_when_marked_invalid(self) -> None:
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "no NaN or Inf"):
                    result(
                        np.array([[invalid]], dtype=np.float32),
                        np.zeros((1, 1), dtype=bool),
                    )

    def test_rejects_mismatched_confidence_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "confidence"):
            BACKENDS.StereoResult(
                disparity_px=np.ones((2, 3), dtype=np.float32),
                valid_mask=np.ones((2, 3), dtype=bool),
                confidence=np.ones((2, 2), dtype=np.float32),
                backend="fake",
                precision="fp32",
                input_shape=(1, 3, 2, 3),
                model_sha256=None,
                timings_ms={"total": 1},
            )


class PinnedOpenStereoPreprocessingTests(unittest.TestCase):
    """Assertions grounded in OpenStereo revision 23d71c9.

    Sources:
      cfgs/lightstereo/lightstereo_s_kitti.yaml
      stereo/datasets/dataset_utils/stereo_trans.py::RightTopPad
      deploy/export.py::export_onnx
    """

    def test_pinned_model_contract_constants(self) -> None:
        self.assertEqual(
            BACKENDS.OPENSTEREO_REVISION,
            "23d71c92e33ad1f80dfc42bf29f5c6a914d38769",
        )
        self.assertEqual(
            BACKENDS.LIGHTSTEREO_CHECKPOINT_SHA256,
            "3d768e0344c2b8bfacb8f7f27cc647cd338e5ba93ec66d944a9a73fd63ec9b2a",
        )
        self.assertEqual(
            BACKENDS.LIGHTSTEREO_CONFIG_RELATIVE.as_posix(),
            "cfgs/lightstereo/lightstereo_s_kitti.yaml",
        )
        self.assertEqual(
            BACKENDS.OPENSTEREO_ONNX_INPUT_NAMES,
            ("left_img", "right_img"),
        )
        self.assertEqual(BACKENDS.OPENSTEREO_ONNX_OUTPUT_NAME, "disp_pred")

    def test_right_top_pad_replicates_edges_and_converts_bgr_to_rgb(self) -> None:
        preprocessor = BACKENDS.LightStereoPreprocessor(target_shape=(3, 3))
        left_bgr = np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ],
            dtype=np.uint8,
        )
        right_bgr = left_bgr + 10

        left, right, pad = preprocessor.prepare(left_bgr, right_bgr)

        self.assertEqual(left.shape, (1, 3, 3, 3))
        self.assertEqual(right.shape, (1, 3, 3, 3))
        self.assertEqual((pad.top, pad.right, pad.bottom, pad.left), (1, 1, 0, 0))
        normalized_hwc = left[0].transpose(1, 2, 0)
        restored_rgb = (
            normalized_hwc * np.asarray(BACKENDS.IMAGENET_STD)
            + np.asarray(BACKENDS.IMAGENET_MEAN)
        ) * 255.0
        expected_rgb = np.pad(
            left_bgr[..., ::-1],
            ((1, 0), (0, 1), (0, 0)),
            mode="edge",
        )
        np.testing.assert_allclose(restored_rgb, expected_rgb, atol=2e-5)

    def test_native_guardian_shape_gets_exactly_24_top_pad_pixels(self) -> None:
        image = np.zeros((360, 640, 3), dtype=np.uint8)
        left, _, pad = BACKENDS.LightStereoPreprocessor().prepare(image, image)

        self.assertEqual(left.shape, BACKENDS.LIGHTSTEREO_INPUT_SHAPE)
        self.assertEqual((pad.top, pad.right), (24, 0))

    def test_restore_crops_padded_output_back_to_native_coordinates(self) -> None:
        preprocessor = BACKENDS.LightStereoPreprocessor()
        image = np.zeros((360, 640, 3), dtype=np.uint8)
        _, _, pad = preprocessor.prepare(image, image)
        padded = np.arange(384 * 640, dtype=np.float32).reshape(1, 1, 384, 640)

        native = preprocessor.restore_disparity(padded, pad)

        self.assertEqual(native.shape, (360, 640))
        np.testing.assert_array_equal(native, padded[0, 0, 24:, :])

    def test_oversize_input_fails_instead_of_resizing(self) -> None:
        image = np.zeros((385, 640, 3), dtype=np.uint8)
        with self.assertRaisesRegex(
            BACKENDS.BackendConfigurationError, "Do not resize silently"
        ):
            BACKENDS.LightStereoPreprocessor().prepare(image, image)

    def test_static_opset17_onnx_contract_is_accepted(self) -> None:
        model = fake_onnx_model(
            opset=17,
            input_shape=BACKENDS.LIGHTSTEREO_INPUT_SHAPE,
            output_shape=(1, 1, 384, 640),
            input_element_type=1,
        )

        BACKENDS.validate_lightstereo_onnx_model(
            model, expected_precision="fp32"
        )

    def test_dynamic_or_wrong_opset_onnx_contract_is_rejected(self) -> None:
        dynamic = fake_onnx_model(
            opset=17,
            input_shape=(1, 3, 0, 640),
            output_shape=(1, 1, 0, 640),
            input_element_type=1,
        )
        wrong_opset = fake_onnx_model(
            opset=18,
            input_shape=BACKENDS.LIGHTSTEREO_INPUT_SHAPE,
            output_shape=(1, 1, 384, 640),
            input_element_type=1,
        )

        with self.assertRaisesRegex(
            BACKENDS.BackendConfigurationError, "static shape"
        ):
            BACKENDS.validate_lightstereo_onnx_model(dynamic)
        with self.assertRaisesRegex(
            BACKENDS.BackendConfigurationError, "opset 17"
        ):
            BACKENDS.validate_lightstereo_onnx_model(wrong_opset)

    def test_declared_onnx_precision_must_match_graph_input_dtype(self) -> None:
        fp32_model = fake_onnx_model(
            opset=17,
            input_shape=BACKENDS.LIGHTSTEREO_INPUT_SHAPE,
            output_shape=(1, 1, 384, 640),
            input_element_type=1,
        )

        with self.assertRaisesRegex(
            BACKENDS.BackendConfigurationError, "declared fp16"
        ):
            BACKENDS.validate_lightstereo_onnx_model(
                fp32_model, expected_precision="fp16"
            )


class BackendConstructionTests(unittest.TestCase):
    def test_sgbm_rejects_non_fp32_precision(self) -> None:
        with self.assertRaisesRegex(
            BACKENDS.BackendConfigurationError, "only supports"
        ):
            BACKENDS.create_backend("sgbm", precision="fp16")

    def test_learned_backend_requires_external_model_artifact(self) -> None:
        with self.assertRaisesRegex(
            BACKENDS.BackendConfigurationError, "--model-path"
        ):
            BACKENDS.create_backend("lightstereo-onnx", precision="fp32")

    def test_optional_dependency_error_is_actionable(self) -> None:
        with mock.patch.object(
            BACKENDS.importlib,
            "import_module",
            side_effect=ImportError("missing wheel"),
        ):
            with self.assertRaisesRegex(
                BACKENDS.OptionalDependencyError, "python -m pip install"
            ):
                BACKENDS._optional_import(
                    "onnxruntime", "python -m pip install onnxruntime-gpu"
                )

    def test_openstereo_revision_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "stereo").mkdir()
            (root / ".git").mkdir()
            completed = subprocess_result(stdout="0" * 40 + "\n")
            with mock.patch.object(
                BACKENDS.subprocess, "run", return_value=completed
            ):
                with self.assertRaisesRegex(
                    BACKENDS.BackendConfigurationError, "revision mismatch"
                ):
                    BACKENDS.verify_openstereo_revision(root)

    def test_openstereo_dirty_tracked_tree_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "stereo").mkdir()
            (root / ".git").mkdir()
            responses = [
                subprocess_result(stdout=BACKENDS.OPENSTEREO_REVISION + "\n"),
                subprocess_result(stdout=" M stereo/modeling/model.py\n"),
            ]
            with mock.patch.object(
                BACKENDS.subprocess, "run", side_effect=responses
            ):
                with self.assertRaisesRegex(
                    BACKENDS.BackendConfigurationError, "tracked tree is dirty"
                ):
                    BACKENDS.verify_openstereo_revision(root)

    def test_official_checkpoint_hash_is_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.ckpt"
            checkpoint.write_bytes(b"not official")
            with self.assertRaisesRegex(
                BACKENDS.BackendConfigurationError, "SHA-256 mismatch"
            ):
                BACKENDS.verify_lightstereo_checkpoint(checkpoint)

    def test_safe_checkpoint_loader_uses_weights_only(self) -> None:
        fake_model = mock.Mock()
        fake_model.eval.return_value.to.return_value = fake_model
        fake_torch = mock.Mock()
        fake_torch.load.return_value = {"model_state": {"weight": object()}}
        fake_timm = mock.Mock()
        config = BACKENDS._ConfigNode(
            {"MODEL": BACKENDS._ConfigNode({"NAME": "LightStereo"})}
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            config_path = root / "config.yaml"
            config_path.write_text("MODEL: {NAME: LightStereo}", encoding="utf-8")
            with (
                mock.patch.object(
                    BACKENDS,
                    "verify_openstereo_revision",
                    return_value=BACKENDS.OPENSTEREO_REVISION,
                ),
                mock.patch.object(
                    BACKENDS,
                    "resolve_pinned_lightstereo_config",
                    return_value=config_path,
                ),
                mock.patch.object(
                    BACKENDS,
                    "verify_lightstereo_checkpoint",
                    return_value=BACKENDS.LIGHTSTEREO_CHECKPOINT_SHA256,
                ),
                mock.patch.object(
                    BACKENDS, "load_lightstereo_config", return_value=config
                ),
                mock.patch.object(
                    BACKENDS,
                    "load_pinned_lightstereo_class",
                    return_value=lambda _: fake_model,
                ),
            ):
                BACKENDS.instantiate_pinned_lightstereo(
                    torch_module=fake_torch,
                    yaml_module=mock.Mock(),
                    timm_module=fake_timm,
                    checkpoint_path=checkpoint,
                    openstereo_root=root,
                    config_path=config_path,
                    device="cpu",
                )

        fake_torch.load.assert_called_once_with(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )

    def test_sgbm_result_implements_common_contract(self) -> None:
        rng = np.random.default_rng(7)
        left = rng.integers(0, 256, size=(100, 160, 3), dtype=np.uint8)
        right = np.roll(left, -4, axis=1)
        backend = BACKENDS.SgbmBackend(opencv_threads=1, stereo_workers=1)
        try:
            output = backend.infer(left, right)
        finally:
            backend.close()

        self.assertEqual(output.backend, "sgbm")
        self.assertEqual(output.disparity_px.shape, (100, 160))
        self.assertEqual(output.valid_mask.shape, (100, 160))
        self.assertIn("stereo_total", output.timings_ms)
        self.assertIn("preprocess_orchestration", output.timings_ms)
        self.assertEqual(output.metadata["opencv_threads"], 1)
        self.assertEqual(output.metadata["stereo_roi_top"], 0)

    def test_sgbm_top_crop_keeps_native_contract_and_invalidates_top(self) -> None:
        rng = np.random.default_rng(8)
        left = rng.integers(0, 256, size=(100, 160, 3), dtype=np.uint8)
        right = np.roll(left, -4, axis=1)
        backend = BACKENDS.SgbmBackend(
            opencv_threads=1,
            stereo_workers=1,
            stereo_roi_top=20,
        )
        try:
            output = backend.infer(left, right)
        finally:
            backend.close()

        self.assertEqual(output.input_shape, (1, 3, 100, 160))
        self.assertEqual(output.disparity_px.shape, (100, 160))
        self.assertFalse(np.any(output.valid_mask[:20]))
        assert output.confidence is not None
        self.assertFalse(np.any(output.confidence[:20]))
        self.assertEqual(output.metadata["stereo_roi_top"], 20)

    def test_sgbm_rejects_negative_top_crop(self) -> None:
        with self.assertRaisesRegex(
            BACKENDS.BackendConfigurationError, "non-negative"
        ):
            BACKENDS.SgbmBackend(stereo_roi_top=-1)

    def test_sgbm_rejects_top_crop_outside_actual_input(self) -> None:
        image = np.zeros((20, 160, 3), dtype=np.uint8)
        backend = BACKENDS.SgbmBackend(
            opencv_threads=1,
            stereo_roi_top=20,
        )
        try:
            with self.assertRaisesRegex(ValueError, r"\[0, 19\]"):
                backend.infer(image, image)
        finally:
            backend.close()


class DisparityParityTests(unittest.TestCase):
    def test_reports_error_and_coverage_without_hiding_invalidations(self) -> None:
        reference = result(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        )
        candidate = result(
            np.array([[1.0, 2.5], [-1.0, 8.5]], dtype=np.float32),
            np.array([[True, True], [False, True]]),
        )

        parity = BACKENDS.disparity_parity(reference, candidate)

        self.assertEqual(parity["compared_pixels"], 3)
        self.assertAlmostEqual(parity["mean_absolute_error_px"], 5.0 / 3.0)
        self.assertAlmostEqual(parity["bad_3px_fraction"], 1.0 / 3.0)
        self.assertAlmostEqual(parity["missing_reference_valid_fraction"], 0.25)

    def test_requires_a_shared_native_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical native shapes"):
            BACKENDS.disparity_parity(
                result(np.ones((2, 2), dtype=np.float32)),
                result(np.ones((3, 2), dtype=np.float32)),
            )


def subprocess_result(*, stdout: str) -> mock.Mock:
    completed = mock.Mock()
    completed.stdout = stdout
    return completed


def fake_value_info(
    name: str, shape: tuple[int, ...], *, element_type: int = 1
) -> types.SimpleNamespace:
    dimensions = [types.SimpleNamespace(dim_value=value) for value in shape]
    tensor_shape = types.SimpleNamespace(dim=dimensions)
    tensor_type = types.SimpleNamespace(
        shape=tensor_shape, elem_type=element_type
    )
    return types.SimpleNamespace(
        name=name,
        type=types.SimpleNamespace(tensor_type=tensor_type),
    )


def fake_onnx_model(
    *,
    opset: int,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    input_element_type: int,
) -> types.SimpleNamespace:
    graph = types.SimpleNamespace(
        input=[
            fake_value_info(
                name, input_shape, element_type=input_element_type
            )
            for name in BACKENDS.OPENSTEREO_ONNX_INPUT_NAMES
        ],
        output=[
            fake_value_info(BACKENDS.OPENSTEREO_ONNX_OUTPUT_NAME, output_shape)
        ],
    )
    return types.SimpleNamespace(
        opset_import=[types.SimpleNamespace(domain="", version=opset)],
        graph=graph,
    )


if __name__ == "__main__":
    unittest.main()
