from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path
from unittest import mock


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
MODULE_PATH = SRC / "classical_tracking.py"
SPEC = importlib.util.spec_from_file_location("classical_tracking", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {MODULE_PATH}")
TRACKING = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRACKING
SPEC.loader.exec_module(TRACKING)

from classical_geometry import ObstacleComponent


def component(x: int, depth: float) -> ObstacleComponent:
    return ObstacleComponent(
        component_id=1,
        x=x,
        y=100,
        width=40,
        height=60,
        area=1200,
        center_x=x + 20,
        center_y=130,
        bottom_y=160,
        depth_m=depth,
        depth_p20_m=depth,
        depth_p35_m=depth,
        depth_mad_m=0.1,
        lr_support=0.9,
        corridor_overlap=1.0,
        quality=0.9,
    )


class TrackingTests(unittest.TestCase):
    def test_same_component_keeps_track_id(self) -> None:
        tracker = TRACKING.ComponentTracker((200, 300))
        ids = []
        for index in range(4):
            tracks = tracker.update(
                [component(120 + index, 10.0 - index * 0.1)],
                index * 0.05,
            )
            ids.append(tracks[0].track_id)
        self.assertEqual(len(set(ids)), 1)
        self.assertTrue(tracks[0].confirmed)

    def test_linear_closing_track_has_expected_ttc(self) -> None:
        tracker = TRACKING.ComponentTracker((200, 300))
        track = None
        for index in range(7):
            tracks = tracker.update(
                [component(120, 10.0 - 2.0 * index * 0.05)],
                index * 0.05,
            )
            track = tracks[0]
        assert track is not None
        closing_speed, ttc, residual = track.motion_state()
        self.assertAlmostEqual(closing_speed, 2.0, places=6)
        self.assertAlmostEqual(ttc, 4.7, places=6)
        self.assertAlmostEqual(residual, 0.0, places=6)

    def test_receding_track_returns_infinite_ttc(self) -> None:
        tracker = TRACKING.ComponentTracker((200, 300))
        for index in range(4):
            tracks = tracker.update(
                [component(120, 5.0 + index * 0.1)],
                index * 0.05,
            )
        self.assertTrue(math.isinf(tracks[0].motion_state()[1]))

    def test_expired_track_is_not_reused(self) -> None:
        tracker = TRACKING.ComponentTracker((200, 300), maximum_missed=1)
        first_id = tracker.update([component(120, 8.0)], 0.0)[0].track_id
        tracker.update([], 0.05)
        tracker.update([], 0.10)
        second_id = tracker.update([component(120, 8.0)], 0.15)[0].track_id
        self.assertNotEqual(first_id, second_id)

    def test_risk_corridor_is_cached_without_changing_selection(self) -> None:
        original_factory = TRACKING.collision_corridor_mask
        with mock.patch.object(
            TRACKING,
            "collision_corridor_mask",
            wraps=original_factory,
        ) as corridor_factory:
            tracker = TRACKING.ComponentTracker((200, 300))
            tracks = []
            for frame_index in range(3):
                tracks = tracker.update(
                    [component(120, 8.0), component(0, 12.0)],
                    frame_index * 0.05,
                )
            first_selection = tracker.risk_tracks(tracks)
            second_selection = tracker.risk_tracks(tracks)

        reference_corridor = original_factory(
            (200, 300),
            top_width_fraction=0.16,
            bottom_width_fraction=0.55,
        )
        expected_ids = [
            track.track_id
            for track in tracks
            if reference_corridor[
                min(199, track.bbox[3] - 1),
                int((track.bbox[0] + track.bbox[2]) / 2),
            ]
        ]
        self.assertEqual(
            [track.track_id for track in first_selection],
            expected_ids,
        )
        self.assertEqual(second_selection, first_selection)
        corridor_factory.assert_called_once()
        self.assertFalse(tracker._risk_corridor.flags.writeable)


if __name__ == "__main__":
    unittest.main()
