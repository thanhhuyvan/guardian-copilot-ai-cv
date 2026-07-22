# Versioning and Releases

The project uses Semantic Versioning: `MAJOR.MINOR.PATCH`.

## During the hackathon

- `0.x.y`: research and prototype development.
- Increment `MINOR` when a verified phase or major capability is completed.
- Increment `PATCH` for backward-compatible fixes, tuning or documentation.
- Use pre-release tags for candidates, for example `v0.4.0-rc.1`.

Suggested milestones:

| Version | Milestone |
|---|---|
| `0.1.0` | Repository and research scaffold |
| `0.2.0` | Dataset audit and reproducible baseline |
| `0.3.0` | Detector/tracker pipeline |
| `0.4.0` | Depth, motion and per-track TTC |
| `0.5.0` | Risk events and integration contract |
| `0.6.0` | Robustness and latency verified |
| `0.9.0` | Submission release candidate |
| `1.0.0` | Final verified hackathon delivery |

## Release checklist

- Update `VERSION`.
- Move relevant `Unreleased` entries in `CHANGELOG.md`.
- Run unit, integration, regression and verification checks.
- Freeze model/config checksums.
- Create an annotated Git tag `vX.Y.Z`.
- Create GitHub Release notes without dataset/model assets that cannot be distributed.

