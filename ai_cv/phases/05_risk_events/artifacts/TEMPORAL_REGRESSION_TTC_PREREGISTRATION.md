# Preregistration — classical temporal-regression TTC experiment

## Motivation

On corrected Phase 17 classical danger frames, false alerts have a median
linear-depth residual MAD of `0.164 m`, versus `0.082 m` for true danger.
Hard association, detector veto, and depth/box-scale agreement have already
been rejected. This experiment tests one causal repair: derive classical TTC
from its existing short track history, rather than the most recent depth
difference.

## Fixed method

For a frame whose finalized V1 union source is classical:

1. use the existing selected classical observations only;
2. require at least five time-ordered finite observations;
3. fit weighted least-squares depth against time, with weight
   `1 / depth_sigma_m^2` where supplied;
4. emit `current_depth / -fitted_depth_rate` when the fitted rate is negative;
5. otherwise emit infinity; and
6. retain V1's final TTC unchanged for all non-classical rows or unavailable
   histories.

No organizer ground truth enters prediction generation. There is no F1-tuned
threshold, fallback, risk gate, FSM change, or parameter sweep.

## Success / failure

Compare the six-trip official report against V1 (`F1 0.654`, critical TTC MAE
`29.993 s`, composite `42.8`). A candidate is promising only if it improves
F1 without increasing critical TTC MAE. Otherwise it is rejected and V1 stays
unchanged.
