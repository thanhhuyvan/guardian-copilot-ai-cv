# Branching and Commit Strategy

## Long-lived branches

- `main`: stable, verified and release-ready.
- `develop`: integration branch for completed phase work.

## Task branches

- `research/phase-XX-topic`: investigation and experiments.
- `feat/component-name`: production feature.
- `fix/short-description`: bug fix.
- `test/short-description`: test or verification work.
- `docs/short-description`: documentation only.
- `chore/short-description`: tooling and repository maintenance.

Examples:

```text
research/phase-01-data-audit
research/phase-02-detector-benchmark
feat/per-track-ttc
fix/reset-track-history
test/submission-validator
```

## Merge policy

- Task branch -> pull request -> `develop`.
- `develop` -> release pull request -> `main`.
- Prefer squash merge for experiment/task branches.
- Never push unverified research directly to `main`.
- Delete merged short-lived branches.

## Conventional Commits

```text
feat(ttc): add per-track closing-speed estimator
fix(depth): reject invalid stereo disparity
test(submission): validate all scored frame ids
research(detector): benchmark candidate models
docs(scope): freeze perception output contract
chore(repo): add GitHub governance files
```

## Commit scope suggestions

- `data`, `detector`, `tracker`, `depth`, `ttc`, `events`
- `robustness`, `latency`, `submission`, `contracts`, `repo`

