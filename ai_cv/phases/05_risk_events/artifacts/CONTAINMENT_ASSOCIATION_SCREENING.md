# Cross-trip containment association screening — not promoted

## Shadow coverage

The no-threshold containment rule proposes an association only when exactly one
YOLO stereo-track centre lies inside a classical component box. Across six
trips it uniquely covers 161/212 classical danger frames (89 true-danger,
72 false-alert). It is therefore useful as an identity-audit signal, not a
risk signal.

## Cross-trip visual screen

The 22-frame stratified pack confirms two opposing facts:

1. T05 broad classical components commonly contain the correct tightly boxed
   pedestrian/car. Hard IoU misses these valid identity links.
2. In a T02 dark false-alert example, a broad classical component covering a
   motorcycle/road region also contains a nearby car. Containment would link
   to that car despite it not explaining the selected classical component.

## Decision

Do not promote containment-only matching or run it through TTC/F1. It would
solve the T05 IoU false negatives by creating false associations in other
conditions.

The remaining association hypothesis is narrower: a temporal, one-to-one
assignment must distinguish component target from nearby road users using
predicted track position and image geometry. Classical component depth must not
be a hard match term, because T05 visually correct pairs have contaminated
component depth. Validate that assignment on labeled cross-trip examples before
changing risk logic.
