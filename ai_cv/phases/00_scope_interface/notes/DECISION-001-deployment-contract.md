# DECISION-001 - Deployment-Neutral Causal Perception Contract

## Status

Accepted for AI/CV research on 2026-07-22.

## Context

The proposal emphasizes an in-car Fast Path, while organizer guidance allows an out-car fleet/post-trip product. The AI/CV core should not be rewritten when the product direction changes.

## Decision

- The primary product narrative is out-car Fleet Collision Intelligence and post-trip analytics.
- The core perception pipeline is causal by default and uses no future frames.
- The same frame/event contracts support in-car streaming and out-car batch processing.
- Deployment-specific latency targets live in runtime config and benchmark reports, not in the perception schema.
- TTC is the required core output; DMS, Safety Kernel, CAN, HMI and product analytics are outside the TTC core.
- JSON uses `null` for no finite TTC; competition CSV serializes it as `inf`.
- File-based CSV/JSON/JSONL is the initial handoff; REST, gRPC or event bus transport is deferred to integration.
- Accuracy research proceeds independently of target hardware; hardware-specific promotion gates are established in Phase 06.

## Consequences

- Online correctness can be evaluated even if the first product demo is post-trip.
- Optional offline smoothing must be explicitly named and cannot silently replace causal predictions.
- Integration can build dashboard/event features without coupling to the chosen CV model.

## Deferred, non-blocking decisions

- Exact target edge/cloud hardware and latency SLA: Phase 06.
- Production transport (REST, gRPC or event bus): integration phase.
