# GuardianCoPilot AI/CV

AI/CV workspace for the GuardianCoPilot collision-risk perception pipeline.

## Objective

Build a reproducible road-perception pipeline that detects and tracks road objects, estimates depth and relative motion, predicts per-frame Time-To-Collision (TTC), and exports risk events and competition submission files.

## Current status

- Project version: `0.1.0` (research scaffold).
- Dataset and starter kit are local-only competition assets and are not committed.
- Research is organized into eight gated phases under [`ai_cv/phases`](ai_cv/phases).

## Repository map

```text
ai_cv/
|- phases/          Phase-specific research, code, tests and verification
|- shared/          Stable contracts, configs and reusable utilities
|- tests/           Cross-phase test suites
|- verification/    Repository and release verification
|- outputs/         Generated local artifacts (ignored by Git)
|- models/          Model metadata; large weights are ignored
|- notebooks/       Exploration only
`- docs/            Architecture, experiments and integration docs
```

## Local data layout

Place competition assets beside `ai_cv/`:

```text
Practice_Dataset/
Hackathon_Dataset_Redacted/
Package_starterkit/
```

Do not commit or upload these directories.

## Start here

1. Read [`AI_CV_WORK_PLAN.md`](AI_CV_WORK_PLAN.md).
2. Read [`ai_cv/README.md`](ai_cv/README.md).
3. Work through [`Phase 00`](ai_cv/phases/00_scope_interface/README.md).
4. Use the branch conventions in [`docs/BRANCHING.md`](docs/BRANCHING.md).

## Verify workspace

```powershell
powershell -ExecutionPolicy Bypass -File .\ai_cv\verification\check_structure.ps1
```

CI uses `-SkipDatasetCheck` because competition assets are intentionally absent from GitHub.

## Governance

- Versioning: [Semantic Versioning](docs/VERSIONING.md).
- Branches and commits: [`docs/BRANCHING.md`](docs/BRANCHING.md).
- Contributions: [`CONTRIBUTING.md`](CONTRIBUTING.md).
- Changelog: [`CHANGELOG.md`](CHANGELOG.md).

