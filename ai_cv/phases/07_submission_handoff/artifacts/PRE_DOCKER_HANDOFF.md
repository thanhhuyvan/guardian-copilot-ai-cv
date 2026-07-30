# Pre-Docker V1 handoff

Docker must package the frozen V1 release, not select a backend or tune a
model. Before building the image, run `preflight_v1_release.py` with
`V1_RUNTIME_CONFIG.json` and `yolo26n.pt`. It verifies:

1. exact model SHA-256;
2. CUDA availability and resolved package versions;
3. the causal run-manifest contract; and
4. the deployed input/output contract versions.

The image must expose causal stereo-pair plus ego-telemetry input and emit
`perception.v1` frames and `risk_event.v1` events. It must preserve the
unreadable-input fail-safe described in `V1_RUNTIME_CONFIG.json`.

TensorRT is deliberately not a pre-Docker requirement. The approved release
backend is PyTorch/CUDA. TensorRT can be assessed later only through a separate
ONNX parity and complete end-to-end latency recertification.
