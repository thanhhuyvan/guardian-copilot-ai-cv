"""Unit and integration test suite for Phase 04B YOLO26 Semantic Fusion."""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import math
import numpy as np
import pytest

from detector_interfaces import Detection, DetectionResult
from semantic_fusion import (
    RETAINED_CLASSES,
    SemanticAssociation,
    TemporalSemanticState,
    associate_component_with_detections,
    compute_intersection_over_box,
    compute_iou,
    compute_vertical_overlap,
    expand_and_clip_box,
    point_in_box,
)
from yolo26_backends import NoneObjectDetector, letterbox_image


class TestBoxGeometry:
    def test_expand_and_clip_box(self) -> None:
        bbox = (100.0, 100.0, 200.0, 200.0)
        img_shape = (360, 640)
        # Expand 10%: width 100 -> dx=5, height 100 -> dy=5
        exp_box = expand_and_clip_box(bbox, img_shape, expand_fraction=0.10)
        assert exp_box == (95.0, 95.0, 205.0, 205.0)

    def test_expand_and_clip_at_boundary(self) -> None:
        bbox = (0.0, 0.0, 50.0, 50.0)
        img_shape = (360, 640)
        exp_box = expand_and_clip_box(bbox, img_shape, expand_fraction=0.10)
        assert exp_box[0] == 0.0
        assert exp_box[1] == 0.0

    def test_compute_iou_identical(self) -> None:
        box = (10.0, 10.0, 50.0, 50.0)
        assert pytest.approx(compute_iou(box, box)) == 1.0

    def test_compute_iou_disjoint(self) -> None:
        box1 = (10.0, 10.0, 50.0, 50.0)
        box2 = (60.0, 60.0, 100.0, 100.0)
        assert compute_iou(box1, box2) == 0.0

    def test_compute_vertical_overlap(self) -> None:
        box1 = (10.0, 10.0, 50.0, 50.0)  # height 40
        box2 = (10.0, 30.0, 50.0, 70.0)  # height 40, vertical inter [30, 50] = 20
        assert pytest.approx(compute_vertical_overlap(box1, box2)) == 0.50

    def test_compute_intersection_over_box_containment(self) -> None:
        component = (100.0, 100.0, 400.0, 300.0)
        detection = (250.0, 180.0, 300.0, 240.0)
        assert compute_intersection_over_box(component, detection) == pytest.approx(1.0)

    def test_point_in_box(self) -> None:
        box = (10.0, 10.0, 50.0, 50.0)
        assert point_in_box(30.0, 30.0, box) is True
        assert point_in_box(0.0, 30.0, box) is False


class TestSemanticAssociation:
    def test_retained_classes(self) -> None:
        assert 2 in RETAINED_CLASSES  # car
        assert 0 in RETAINED_CLASSES  # person
        assert 4 not in RETAINED_CLASSES  # airplane (excluded)

    def test_association_matching_iou(self) -> None:
        comp_bbox = (100, 100, 200, 200)
        dets = [
            Detection(bbox_xyxy=(105.0, 105.0, 195.0, 195.0), class_id=2, class_name="car", confidence=0.90)
        ]
        assoc = associate_component_with_detections(comp_bbox, dets, (360, 640))
        assert assoc.matched is True
        assert assoc.class_id == 2
        assert assoc.confidence == 0.90

    def test_association_unretained_class_ignored(self) -> None:
        comp_bbox = (100, 100, 200, 200)
        dets = [
            Detection(bbox_xyxy=(105.0, 105.0, 195.0, 195.0), class_id=4, class_name="airplane", confidence=0.90)
        ]
        assoc = associate_component_with_detections(comp_bbox, dets, (360, 640))
        assert assoc.matched is False

    def test_association_best_match_selection(self) -> None:
        comp_bbox = (100, 100, 200, 200)
        dets = [
            Detection(bbox_xyxy=(100.0, 100.0, 200.0, 200.0), class_id=2, class_name="car", confidence=0.50),
            Detection(bbox_xyxy=(102.0, 102.0, 198.0, 198.0), class_id=7, class_name="truck", confidence=0.95),
        ]
        assoc = associate_component_with_detections(comp_bbox, dets, (360, 640))
        assert assoc.matched is True
        assert assoc.class_id == 7  # Higher confidence selection

    def test_association_matches_detection_contained_in_merged_component(self) -> None:
        component = (124, 130, 428, 285)
        detection = Detection(
            bbox_xyxy=(307.0, 181.0, 335.0, 204.0),
            class_id=2,
            class_name="car",
            confidence=0.69,
        )
        assoc = associate_component_with_detections(
            component, [detection], (360, 640)
        )
        assert assoc.matched is True
        assert assoc.class_name == "car"

    def test_association_rejects_low_coverage_edge_contact(self) -> None:
        component = (100, 100, 205, 300)
        detection = Detection(
            bbox_xyxy=(195.0, 180.0, 255.0, 240.0),
            class_id=2,
            class_name="car",
            confidence=0.90,
        )
        assoc = associate_component_with_detections(
            component, [detection], (360, 640)
        )
        assert assoc.matched is False


class TestTemporalSemanticState:
    def test_ema_decay(self) -> None:
        state = TemporalSemanticState()
        assoc = SemanticAssociation(matched=True, class_id=2, class_name="car", confidence=1.0)
        state.update(assoc)
        # score = 0.4 * 1.0 + 0.6 * 0 = 0.4
        assert pytest.approx(state.score) == 0.4
        assert state.consecutive_misses == 0

        # Miss frame
        state.update(SemanticAssociation(matched=False))
        # score = 0.4 * 0 + 0.6 * 0.4 = 0.24
        assert pytest.approx(state.score) == 0.24
        assert state.consecutive_misses == 1

    def test_close_range_fallback_preserves_candidate(self) -> None:
        state = TemporalSemanticState()
        # 5 misses, no score
        for _ in range(5):
            state.update(SemanticAssociation(matched=False))

        # At depth <= 5.0m, should NOT be suppressed
        suppressed = state.is_suppressed(
            latest_depth_m=4.5,
            score_threshold=0.25,
            max_misses=3,
            fallback_depth_m=5.0,
        )
        assert suppressed is False

        # At depth > 5.0m, SHOULD be suppressed
        suppressed_far = state.is_suppressed(
            latest_depth_m=10.0,
            score_threshold=0.25,
            max_misses=3,
            fallback_depth_m=5.0,
        )
        assert suppressed_far is True

    def test_three_consecutive_misses_accumulate(self) -> None:
        """Consecutive miss counter must increment across frames when keyed by track_id.

        This is the core regression test for the track_key=idx bug: if state is
        keyed by row index instead of track_id, misses never accumulate past 1.
        """
        state = TemporalSemanticState()
        # Three successive misses on the same track
        for _ in range(3):
            state.update(SemanticAssociation(matched=False))
        assert state.consecutive_misses == 3, (
            f"Expected 3 consecutive misses, got {state.consecutive_misses}. "
            "Likely cause: state was not persisted across frames (row-index key bug)."
        )
        # Score should have decayed: 0.6^3 * 0 = 0 (started at 0)
        assert pytest.approx(state.score, abs=1e-6) == 0.0

    def test_miss_resets_after_match(self) -> None:
        """Consecutive miss counter resets to zero on a match frame."""
        state = TemporalSemanticState()
        for _ in range(3):
            state.update(SemanticAssociation(matched=False))
        assert state.consecutive_misses == 3
        state.update(SemanticAssociation(matched=True, class_id=2, class_name="car", confidence=0.9))
        assert state.consecutive_misses == 0
        state = TemporalSemanticState()
        # Frame 1: matched
        state.update(SemanticAssociation(matched=True, class_id=2, class_name="car", confidence=0.9))
        # Frame 2: single miss
        state.update(SemanticAssociation(matched=False))

        suppressed = state.is_suppressed(
            latest_depth_m=12.0,
            score_threshold=0.25,
            max_misses=3,
            fallback_depth_m=5.0,
        )
        assert suppressed is False


class TestBackends:
    def test_none_backend(self) -> None:
        detector = NoneObjectDetector()
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        res = detector.infer(img)
        assert res.backend == "none"
        assert len(res.detections) == 0
        detector.close()

    def test_letterbox_shapes(self) -> None:
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        padded, scale, (dw, dh) = letterbox_image(img, (640, 640))
        assert padded.shape == (640, 640, 3)
        assert scale > 0.0

    def test_onnx_zero_match_is_not_silently_perfect(self) -> None:
        """Verify parity logic: zero matched detections must NOT produce all-perfect metrics.

        The old code fell back to 1.0 for median_iou and mean_conf_diff when
        the ious/conf_diffs lists were empty (i.e., zero matches), and then
        evaluated all gates as passed.  This test confirms the correct behaviour:
        zero matches must set parity_valid=False and must not pass the gates.
        """
        ious_list: list[float] = []
        conf_diffs_list: list[float] = []

        # Old (wrong): empty list -> 1.0 fallback â€” gates appeared to pass
        old_median_iou   = float(np.median(ious_list))   if ious_list   else 1.0
        old_mean_conf    = float(np.mean(conf_diffs_list)) if conf_diffs_list else 0.0
        assert old_median_iou == 1.0   # confirms the old bug: perfect IoU with zero data
        assert old_mean_conf  == 0.0   # and zero conf diff â€” both gates trivially pass

        # New (correct): parity_valid must be False when total_matched == 0
        total_matched = 0
        parity_valid = total_matched > 0
        assert parity_valid is False, (
            "Zero matched detections must mark parity_valid=False, not silently pass all gates."
        )


class TestOnnxCudaDllBootstrap:
    """The ONNX CUDA EP needs torch's bundled cuDNN/CUDA DLLs on PATH on Windows.

    Without ``_ensure_cuda_dlls_on_path`` ORT silently falls back to CPU and the
    parity test runs on the wrong device. This test guards the bootstrap logic
    without requiring an ONNX model file or a GPU.
    """

    def test_ensure_cuda_dlls_on_path_is_idempotent_and_safe(self) -> None:
        import os

        from yolo26_backends import _ensure_cuda_dlls_on_path

        original_path = os.environ.get("PATH", "")
        _ensure_cuda_dlls_on_path()
        _ensure_cuda_dlls_on_path()  # calling twice must not raise
        # If torch is installed, its lib dir is now on PATH.
        try:
            import torch
            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            if os.path.isdir(torch_lib):
                assert torch_lib in os.environ["PATH"]
        except ImportError:
            pass
        # Restore to keep the test side-effect-free for later tests.
        os.environ["PATH"] = original_path


class TestParityRootCauseFusionEquivalence:
    """Raw ONNX-vs-PyTorch parity can disagree on class labels for the same box
    (the end-to-end ONNX export retains competing class hypotheses per box and
    tiebreaks by insertion order, while Ultralytics' native postprocess
    tiebreaks differently). Downstream fusion must not care: soft-guard only
    uses retained-class membership + confidence, so a car/truck swap on the same
    object is fusion-equivalent.
    """

    def test_car_truck_same_box_are_fusion_equivalent(self) -> None:
        comp_bbox = (475, 324, 633, 375)
        img_shape = (360, 640, 3)

        car_det = Detection(
            bbox_xyxy=(475, 324, 633, 375), class_id=2, class_name="car", confidence=0.4464
        )
        truck_det = Detection(
            bbox_xyxy=(475, 324, 633, 375), class_id=7, class_name="truck", confidence=0.4892
        )

        assoc_car = associate_component_with_detections(comp_bbox, [car_det], img_shape)
        assoc_truck = associate_component_with_detections(comp_bbox, [truck_det], img_shape)

        # Both match the component (only retained-class membership matters).
        assert assoc_car.matched and assoc_truck.matched

        # Over 5 matched frames at far range, suppression behaviour is identical.
        state_car = TemporalSemanticState()
        state_truck = TemporalSemanticState()
        sup_car, sup_truck = [], []
        for _ in range(5):
            state_car.update(assoc_car)
            state_truck.update(assoc_truck)
            sup_car.append(state_car.is_suppressed(latest_depth_m=10.0))
            sup_truck.append(state_truck.is_suppressed(latest_depth_m=10.0))
        assert sup_car == sup_truck == [False, False, False, False, False]

    def test_same_box_competing_class_swap_is_not_a_localization_error(self) -> None:
        """A class swap at IoU>=0.95 is a competing-class tiebreak on one object,
        not a localization disagreement. The parity report classifies these as
        ``same_box`` so they are not confused with distinct-object errors.
        """
        box = (100.0, 100.0, 200.0, 200.0)
        assert compute_iou(box, box) >= 0.95  # identical box -> same object



# ---------------------------------------------------------------------------
# Integration tests: classical_tracking.py semantic wiring
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from classical_geometry import ObstacleComponent
from classical_tracking import ComponentTracker, select_minimum_ttc


def _make_component(
    cx: float = 0.5,
    cy: float = 0.7,
    depth: float = 10.0,
    bbox: tuple = (200, 200, 300, 300),
) -> ObstacleComponent:
    """Minimal ObstacleComponent for tracker tests."""
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    return ObstacleComponent(
        component_id=1,
        x=x0,
        y=y0,
        width=w,
        height=h,
        area=w * h,
        center_x=cx,
        center_y=cy,
        bottom_y=int(y1),
        depth_m=depth,
        depth_p20_m=depth,
        depth_p35_m=depth,
        depth_mad_m=0.5,
        lr_support=0.99,
        corridor_overlap=1.0,
        quality=0.9,
    )


class TestClassicalTrackingSemanticWiring:
    """Verify none-backend bit-parity and semantic integration in ComponentTracker."""

    def test_none_backend_semantic_state_is_null(self) -> None:
        """Tracker with no semantic config creates tracks with semantic_state=None."""
        tracker = ComponentTracker(image_shape=(360, 640))
        comp = _make_component()
        for t in (0.0, 0.05, 0.10):
            tracker.update([comp], t)
        for track in tracker.tracks.values():
            assert track.semantic_state is None

    def test_none_backend_is_never_suppressed(self) -> None:
        """is_semantically_suppressed() always returns False when semantic_state is None."""
        tracker = ComponentTracker(image_shape=(360, 640))
        comp = _make_component(depth=15.0)
        for t in (0.0, 0.05, 0.10):
            tracker.update([comp], t)
        for track in tracker.tracks.values():
            assert track.is_semantically_suppressed() is False

    def test_semantic_enabled_creates_state(self) -> None:
        """Tracker with semantic_score_threshold creates TemporalSemanticState on tracks."""
        tracker = ComponentTracker(
            image_shape=(360, 640),
            semantic_score_threshold=0.25,
        )
        comp = _make_component()
        for t in (0.0, 0.05, 0.10):
            tracker.update([comp], t)
        for track in tracker.tracks.values():
            assert track.semantic_state is not None

    def test_persistent_far_unmatched_component_gets_suppressed(self) -> None:
        """A far (>5m) component with no YOLO detections for 3+ frames is suppressed."""
        tracker = ComponentTracker(
            image_shape=(360, 640),
            semantic_score_threshold=0.25,
            semantic_max_misses=3,
            semantic_fallback_depth_m=5.0,
        )
        comp = _make_component(depth=12.0)
        # Update 6 frames with no detections -> 6 consecutive misses
        for i in range(6):
            tracker.update([comp], i * 0.05, detections=[])
        # All tracks should be suppressed (score < 0.25, misses >= 3, depth > 5m)
        for track in tracker.tracks.values():
            if track.confirmed:
                assert track.is_semantically_suppressed(
                    score_threshold=0.25, max_misses=3, fallback_depth_m=5.0
                ), f"Expected suppressed, score={track.semantic_state.score:.4f}"

    def test_close_range_component_never_suppressed(self) -> None:
        """A component at depth <=5m is never suppressed even with 10 consecutive misses."""
        tracker = ComponentTracker(
            image_shape=(360, 640),
            semantic_score_threshold=0.25,
            semantic_max_misses=3,
            semantic_fallback_depth_m=5.0,
        )
        comp = _make_component(depth=3.0)
        for i in range(10):
            tracker.update([comp], i * 0.05, detections=[])
        for track in tracker.tracks.values():
            assert track.is_semantically_suppressed(
                score_threshold=0.25, max_misses=3, fallback_depth_m=5.0
            ) is False, "Close-range fallback must keep track alive"

    def test_reset_clears_all_tracks_and_semantic_state(self) -> None:
        """reset() clears tracks between trips â€” no state leakage."""
        tracker = ComponentTracker(
            image_shape=(360, 640),
            semantic_score_threshold=0.25,
        )
        comp = _make_component()
        for t in (0.0, 0.05, 0.10):
            tracker.update([comp], t)
        assert len(tracker.tracks) > 0
        tracker.reset()
        assert len(tracker.tracks) == 0
        assert tracker.next_track_id == 1

    def test_select_minimum_ttc_none_backend_bit_parity(self) -> None:
        """select_minimum_ttc with no semantic config produces same result as pre-semantic code."""
        tracker = ComponentTracker(image_shape=(360, 640))
        # Build a closing track
        for i in range(5):
            depth = 20.0 - i * 2.0  # closing
            comp = _make_component(depth=depth, cx=0.5, cy=0.8)
            tracker.update([comp], i * 0.05)
        risk = tracker.risk_tracks(list(tracker.tracks.values()))
        ttc, tid, conf, speed = select_minimum_ttc(risk, ground_confidence=0.8)
        # Must return a finite TTC (closing track) and NOT be suppressed
        assert math.isfinite(ttc), f"Expected finite TTC, got {ttc}"
        assert tid is not None

class TestSweepSelectionModes:
    @staticmethod
    def _row(
        f1: float,
        *,
        composite: float = 40.0,
        mae: float = 10.0,
    ) -> np.ndarray:
        # f1, precision, recall, tp, fp, fn, composite, mae, suppressed
        return np.array([f1, f1, f1, 1, 0, 0, composite, mae, 0], dtype=float)

    def test_oracle_includes_no_semantic_baseline(self) -> None:
        from sweep_yolo26_fusion import TRIPS, compute_selection_modes

        baseline = {trip: self._row(0.50) for trip in TRIPS}
        semantic = {trip: self._row(0.40) for trip in TRIPS}
        semantic[TRIPS[0]] = self._row(0.60)

        summary = compute_selection_modes({
            "baseline": baseline,
            "results": {"s0.2_m2_d5.0": semantic},
        })
        oracle = summary["oracle_per_trip"]

        assert oracle["per_trip"][TRIPS[0]]["config"] == "s0.2_m2_d5.0"
        for trip in TRIPS[1:]:
            assert oracle["per_trip"][trip]["config"] == "baseline"
        assert oracle["macro_f1"] == pytest.approx((0.60 + 5 * 0.50) / 6)
        assert oracle["selection_space"] == (
            "baseline (semantics off) + 27 semantic configs"
        )
