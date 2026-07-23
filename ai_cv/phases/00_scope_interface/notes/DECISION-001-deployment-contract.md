# DECISION-001 - Deployment-Neutral Causal Perception Contract

## Status

Accepted for AI/CV research on 2026-07-22.

## Context

The proposal emphasizes an in-car Fast Path, while organizer guidance allows an out-car fleet/post-trip product. The AI/CV core should not be rewritten when the product direction changes.

## Decision

- The primary product narrative is out-car Fleet Collision Intelligence and post-trip analytics.
- The core perception pipeline is causal by default and uses neither future frames nor the future schedule in `events_log`.
- Every run declares either `causal_online` or `offline_post_trip` in a versioned run manifest.
- The same frame/event contracts support in-car streaming and out-car batch processing.
- Deployment-specific latency targets live in runtime config and benchmark reports, not in the perception schema.
- TTC is the required core output; DMS, Safety Kernel, CAN, HMI and product analytics are outside the TTC core.
- JSON uses `null` for no finite TTC; competition CSV serializes it as `inf`.
- File-based CSV/JSON/JSONL is the initial handoff; REST, gRPC or event bus transport is deferred to integration.
- Accuracy research proceeds independently of target hardware, but every promoted experiment records measured P50/P95 latency, throughput and hardware; hardware-specific hard gates are established in Phase 06.
- Depth keyframes require an explicit policy (`validation_only`, `calibration`, `direct_inference` or `interpolation`) in the run manifest.

## Consequences

- Online correctness can be evaluated even if the first product demo is post-trip.
- Optional offline smoothing must be explicitly named and cannot silently replace causal predictions.
- Integration can build dashboard/event features without coupling to the chosen CV model.

## Deferred, non-blocking decisions

- Exact target edge/cloud hardware and production latency SLA: Phase 06.
- Production transport (REST, gRPC or event bus): integration phase.
- Direct/interpolated depth-keyframe use in a final submission: organizer confirmation before promotion.
