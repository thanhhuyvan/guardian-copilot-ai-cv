# DECISION-001 - Deployment-Neutral Causal Perception Contract

## Status

Provisional - requires CV Owner and integration-owner confirmation.

## Context

The proposal emphasizes an in-car Fast Path, while organizer guidance allows an out-car fleet/post-trip product. The AI/CV core should not be rewritten when the product direction changes.

## Decision

- The core perception pipeline is causal by default and uses no future frames.
- The same frame/event contracts support in-car streaming and out-car batch processing.
- Deployment-specific latency targets live in runtime config and benchmark reports, not in the perception schema.
- TTC is the required core output; DMS and product analytics remain auxiliary consumers/producers.
- JSON uses `null` for no finite TTC; competition CSV serializes it as `inf`.

## Consequences

- Online correctness can be evaluated even if the first product demo is post-trip.
- Optional offline smoothing must be explicitly named and cannot silently replace causal predictions.
- Integration can build dashboard/event features without coupling to the chosen CV model.

## Confirmation needed

- Primary product narrative: in-car or out-car.
- Target hardware and latency target.
- Integration transport: file, REST, event bus or gRPC.

