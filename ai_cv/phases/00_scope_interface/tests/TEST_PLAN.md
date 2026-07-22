# Test Plan - Phase 00

- Validate JSON Schema Draft 2020-12 syntax and all positive examples.
- Reject missing/extra fields, malformed hashes and causal future leakage.
- Reject reversed/out-of-bounds bbox, invalid event ordering and finite TTC without closing motion.
- Reject mismatched frame minimum TTC, risk severity and confidence level.
- Confirm TTC at contact (`0`) is valid and maps to `CRITICAL`.
- Run the suite in CI on pull requests and protected branches.
