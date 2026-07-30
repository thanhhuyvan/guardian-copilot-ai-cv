# Looming / image-expansion cue shadow result — rejected

## Question

Can a causal image-scale expansion TTC (looming/tau) safely corroborate the
frozen V1 stereo TTC before a danger event is emitted?

## Fixed shadow method

For each frozen V1 conservative-union danger frame (`TTC < 2 s`), reconstruct
a short history from the selected union YOLO box. Use 3–5 consecutive box
areas, reset after a gap above 0.11 s, and calculate tau from log-area growth.
The two TTC cues agree only when their ratio is at most **2.0**. The method
never changed a prediction.

## Result

| Group | Cue agree | Cue disagree | Unavailable |
|---|---:|---:|---:|
| True V1-danger frames | 48 | 51 | — |
| False V1-danger frames | 19 | 22 | — |
| All V1-danger frames | 67 | 73 | 135 |

Finite looming was available for only **140 / 275 = 50.9%** of V1 danger
frames. More importantly, its disagreement rate is effectively the same for
true and false danger (51.5% versus 53.7%). A gate requiring agreement would
therefore suppress genuine danger at about the same rate as false danger.

## Decision

Reject looming as a V1/V2 TTC gate. Do not run a framewise F1 gate, ratio
sweep, or threshold search: the fixed shadow evidence fails both coverage and
class-separation requirements. The cue may remain diagnostic telemetry if a
future object-level labelled dataset establishes stable object tracks and
camera-motion compensation.

## Reproduction

`src/audit_looming_cue.py` writes the ignored raw report under
`ai_cv/outputs/phase28_looming_shadow/report.json`.
