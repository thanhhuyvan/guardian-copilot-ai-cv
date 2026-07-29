from __future__ import annotations

import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SETUP_ROOT = REPO_ROOT / "scripts" / "phase02b"
SETUP_DOC = REPO_ROOT / "docs" / "PHASE_02B_WSL_SETUP.md"
OPENSTEREO_COMMIT = "23d71c92e33ad1f80dfc42bf29f5c6a914d38769"
CHECKPOINT_SHA256 = (
    "3d768e0344c2b8bfacb8f7f27cc647cd"
    "338e5ba93ec66d944a9a73fd63ec9b2a"
)
SHARED_NUMPY_PIN = "numpy==1.26.4"
SHARED_OPENCV_PIN = "opencv-python-headless==4.11.0.86"


class Phase02BSetupAssetTests(unittest.TestCase):
    def read_setup_file(self, relative_path: str) -> str:
        path = SETUP_ROOT / relative_path
        self.assertTrue(path.is_file(), f"Missing setup asset: {path}")
        return path.read_text(encoding="utf-8")

    def read_setup_doc(self) -> str:
        self.assertTrue(SETUP_DOC.is_file(), f"Missing setup guide: {SETUP_DOC}")
        return SETUP_DOC.read_text(encoding="utf-8")

    def test_openstereo_clone_is_external_and_pinned(self) -> None:
        script = self.read_setup_file("clone_openstereo.sh")
        self.assertIn(
            'OPENSTEREO_URL="https://github.com/XiandaGuo/OpenStereo.git"',
            script,
        )
        self.assertIn(f'OPENSTEREO_COMMIT="{OPENSTEREO_COMMIT}"', script)
        self.assertIn('target_root="${HOME}/benchmarks/OpenStereo"', script)
        self.assertIn("switch --detach", script)
        self.assertIn("findmnt", script)

    def test_wsl_defaults_to_d_drive_and_requires_apply(self) -> None:
        script = self.read_setup_file("bootstrap_wsl.ps1")
        self.assertIn(
            '[string]$InstallLocation = "D:\\WSL\\Ubuntu-22.04"',
            script,
        )
        self.assertIn("[switch]$Apply", script)
        self.assertIn("--no-distribution", script)
        self.assertIn("--location", script)
        self.assertNotIn("--unregister", script)

    def test_cuda_bootstrap_never_requests_a_display_driver(self) -> None:
        script = self.read_setup_file("bootstrap_wsl.sh")
        self.assertIn(
            'CUDA_TOOLKIT_PACKAGE="cuda-toolkit-12-8=12.8.1-1"',
            script,
        )
        self.assertNotRegex(
            script,
            re.compile(r"apt-get install[^\n]*nvidia-driver", re.IGNORECASE),
        )
        self.assertNotRegex(
            script,
            re.compile(r"apt-get install[^\n]*\bcuda(?:-12-8)?(?:\s|$)"),
        )
        self.assertIn("grep -Eq '^nvidia-driver(-|$)'", script)

    def test_python_environments_are_separate_and_pinned(self) -> None:
        bootstrap = self.read_setup_file("bootstrap_wsl.sh")
        guardian = self.read_setup_file("requirements/guardian-py310.txt")
        openstereo = self.read_setup_file(
            "requirements/openstereo-py310.txt"
        )

        self.assertIn('guardian-phase02b"', bootstrap)
        self.assertIn('openstereo-phase02b"', bootstrap)
        self.assertIn('PYTORCH_VERSION="2.7.1"', bootstrap)
        for shared_pin in (SHARED_NUMPY_PIN, SHARED_OPENCV_PIN):
            self.assertIn(shared_pin, guardian)
            self.assertIn(shared_pin, openstereo)
        self.assertIn("nvidia-ml-py==12.570.172", openstereo)
        self.assertIn("onnxruntime-gpu==1.22.0", openstereo)
        self.assertIn("tensorrt-cu12==10.8.0.43", openstereo)
        self.assertIn("timm==0.4.12", openstereo)

        for requirements in (guardian, openstereo):
            for line in requirements.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                self.assertRegex(
                    stripped,
                    r"^[A-Za-z0-9_.-]+==[A-Za-z0-9_.+-]+$",
                    f"Dependency is not exactly pinned: {stripped}",
                )

    def test_shared_numeric_runtime_and_wsl_revalidation_are_enforced(
        self,
    ) -> None:
        smoke = self.read_setup_file("smoke_check.sh")
        guide = self.read_setup_doc()

        for expected in (
            'SHARED_NUMPY_VERSION="1.26.4"',
            'SHARED_OPENCV_PACKAGE_VERSION="4.11.0.86"',
            "version('numpy') == '${SHARED_NUMPY_VERSION}'",
            (
                "version('opencv-python-headless') == "
                "'${SHARED_OPENCV_PACKAGE_VERSION}'"
            ),
        ):
            self.assertIn(expected, smoke)

        for expected in (
            "all 3,600 practice frames",
            "composite `28.7`",
            "danger-F1 `0.402`",
            "~/guardian-data/phase02b/Practice_Dataset",
            "lightstereo_parity_72.json",
        ):
            self.assertIn(expected, guide)

    def test_checkpoint_fetch_is_external_and_checksum_locked(self) -> None:
        script = self.read_setup_file("fetch_openstereo_checkpoint.sh")
        self.assertIn(
            "checkpoint/LightStereo/LightStereo-S-KITTI.ckpt",
            script,
        )
        self.assertIn('CHECKPOINT_SIZE_BYTES="14159749"', script)
        self.assertIn(f'CHECKPOINT_SHA256="{CHECKPOINT_SHA256}"', script)
        self.assertIn(
            'destination="${HOME}/benchmarks/OpenStereo-assets/',
            script,
        )
        self.assertIn("9p|drvfs|fuseblk", script)
        self.assertIn("Refusing to overwrite", script)

    def test_data_staging_requires_ext4_destination_and_never_deletes(self) -> None:
        script = self.read_setup_file("stage_data.sh")
        self.assertIn('destination_root="${HOME}/guardian-data/phase02b"', script)
        self.assertIn("findmnt", script)
        self.assertIn("9p|drvfs|fuseblk", script)
        self.assertIn("--checksum", script)
        self.assertNotIn("--delete", script)

    def test_smoke_gate_covers_gpu_runtimes_and_source_pin(self) -> None:
        script = self.read_setup_file("smoke_check.sh")
        for expected in (
            "nvidia-smi",
            "/usr/local/cuda-12.8/bin/nvcc",
            "torch.cuda.is_available",
            "torch.nn.Conv2d",
            "CUDAExecutionProvider",
            '"session.disable_cpu_ep_fallback", "1"',
            "session.run(",
            "pynvml.nvmlInit",
            "trt.Builder",
            "trtexec",
            "helper.make_opsetid(\"\", 17)",
            "LightStereo-S KITTI checkpoint",
            "lightstereo_runtime_smoke.py",
            OPENSTEREO_COMMIT,
        ):
            self.assertIn(expected, script)

    def test_lightstereo_smoke_uses_real_guardian_backend_and_cuda_path(
        self,
    ) -> None:
        script = self.read_setup_file("lightstereo_runtime_smoke.py")
        for expected in (
            "lightstereo_s_kitti.yaml",
            "EXPECTED_MODEL_CONFIG",
            "LightStereoPyTorchBackend",
            "checkpoint_path=checkpoint_path",
            'device="cuda:0"',
            "backend.infer(left, right)",
            "np.zeros((360, 640, 3)",
            "result.disparity_px.shape != (360, 640)",
            "FoundationStereo/flash-attn",
            OPENSTEREO_COMMIT,
        ):
            self.assertIn(expected, script)


if __name__ == "__main__":
    unittest.main()
