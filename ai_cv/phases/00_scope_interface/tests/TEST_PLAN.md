# Test Plan - Phase 00

- Validate JSON Schema Draft 2020-12 syntax and all positive examples.
- Reject missing/extra fields, malformed hashes and causal future leakage.
- Reject reversed/out-of-bounds bbox, invalid event ordering, mismatched frame minimum TTC and non-finite numbers.
- Allow direct image-space TTC without metric distance/closing speed and stateful risk during hysteresis.
- Reject causal future flags and future depth interpolation.
- Confirm TTC at contact (`0`) is valid; nominal physical severity is `CRITICAL`.
- Run the suite in CI on pull requests and CI target branches.
