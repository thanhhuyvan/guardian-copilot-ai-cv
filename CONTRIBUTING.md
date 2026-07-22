# Contributing

## Workflow

1. Start from the latest `develop`.
2. Create one focused branch using `type/short-description`.
3. Keep commits small and use Conventional Commits.
4. Add or update tests and verification evidence.
5. Open a pull request into `develop`.
6. Merge `develop` into `main` only for verified release candidates.

## Required pull-request evidence

- Problem and scope.
- Approach and alternatives considered.
- Tests executed.
- Metric/latency effect where relevant.
- Output examples or screenshots for visual changes.
- Known limitations and rollback plan.

## Data policy

Never commit competition datasets, raw clips, model weights, credentials or generated submission artifacts. Store only metadata, checksums, configs and reproducible download/build instructions.

