# Temporal-regression TTC result — rejected

The preregistered offline candidate replaced only classical-source V1 TTC with
a weighted linear fit across its existing five-or-more depth observations.
No organizer truth was used for generation; 347 classical rows were replaced
and 3,190 rows retained V1 due to unavailable/non-classical history.

| Metric | V1 | Temporal regression | Change |
|---|---:|---:|---:|
| Overall F1 | 0.654 | 0.633 | -0.021 |
| Critical TTC MAE | 29.993 s | 30.170 s | +0.177 s |
| Composite | 42.8 | 41.9 | -0.9 |
| T05 F1 | 0.509 | 0.491 | -0.018 |

This fails the preregistered success rule: F1 declined and critical TTC MAE
increased. Do not promote this estimator, tune its window/weights, or run
further variants. The result also explains why a simple depth-only filter is
not a sufficient causal repair: smoothing changes genuine closing events as
well as noisy false alerts.

V1 remains the official result. The unresolved next question is geometric:
whether a visible adjacent/diverging road user should be excluded by a
validated ego-path relation, independently of its depth-rate TTC.
