# Stage 2A Classical Component Sequence

Each component is evaluated independently before being composed with later
components. A visually cleaner output is not sufficient; every promoted change
must pass a full TTC ablation or have an explicit diagnostic-only role.

1. **SGBM confidence and LR consistency — COMPLETE**
   - Decision: confidence signal only; hard mask rejected.
2. **Ground-plane/V-disparity removal — COMPLETE / KEEP**
   - Road support is removed while six measured small/lead-object cases remain
     observable.
3. **Obstacle components or Stixel-lite — COMPLETE / DIAGNOSTIC ONLY**
   - Vertical support removes a key T03 road-band failure and recovers small
     objects, but components merge and fragment.
4. **Causal temporal association — COMPLETE / REPLACE IDENTITY SOURCE**
   - Per-track TTC improves recall, but selected ID switches remain high on
     T02/T03/T06.
5. **Robust per-track state — COMPLETE / KEEP**
   - Theil-Sen distance motion and confidence gates are causal and tested.
6. **Collision corridor — COMPLETE / KEEP**
   - Wide extraction and narrow risk corridors separate proposal recall from
     threat selection.
7. **Stage 2B instance-aware extractor — NEXT**
   - Compare a lightweight object detector/instance method with both Stage 1
     robust ROI and Stage 2A track-p35 references.
8. **Optical expansion/flow fallback — CONDITIONAL**
   - Test only if stereo remains the measured bottleneck after identity is
     improved.
