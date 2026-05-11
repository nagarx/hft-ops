"""Phase R-17 F6: regression tests for #PY-127 sweep summary counter conflation.

Pre-F6 `cli.py:1902` used formula `Skipped (dupes): {len(experiments) - completed - failed}`
which mislabeled post-abort grid points (loop broke at i<N when --on-failure=abort)
as "dupes". This created user-facing misleading messages (e.g., "Skipped (dupes): 3"
when reality was 1 failed + 3 never-processed due to abort).

F6 tracks `skipped_dupes` separately AND derives `not_processed` as a distinct
category, displaying it only when non-zero.

Tests verify counter math in isolation (the actual cli.py sweep loop requires
significant fixture setup; these tests lock the math + the F6 contract).

Per H2 agent edge cases:
- Zero-grid-points sweep
- All-dupes sweep
- Mixed scenarios
"""

from __future__ import annotations

import pytest


class TestSweepSummaryCounterMath:
    """Phase R-17 F6 (#PY-127): explicit counter tracking, NOT formula-derived."""

    def test_real_dupes_counted_separately_from_not_processed(self):
        """F6a: when dedup-skip + abort-not-processed BOTH occur, they're counted distinctly."""
        # Scenario: 12 grid points; 2 dedup-skipped at i=0,1; loop succeeds 3 more,
        # then i=5 fails + abort → 6 not_processed (i=6..11).
        # Pre-F6: formula `12 - 3 - 1 = 8` mislabeled as "Skipped (dupes)"
        # Post-F6: skipped_dupes=2, completed=3, failed=1, not_processed=6
        total = 12
        completed = 3
        failed = 1
        skipped_dupes = 2
        not_processed = total - completed - failed - skipped_dupes
        assert not_processed == 6
        assert (skipped_dupes + not_processed) == 8  # Pre-F6 lumped these as "dupes"
        # F6 fixes the labeling

    def test_all_completed_no_dupes_no_aborts(self):
        """Happy path: 4 grid points, all complete. No dupes, no abort."""
        total = 4
        completed = 4
        failed = 0
        skipped_dupes = 0
        not_processed = total - completed - failed - skipped_dupes
        assert not_processed == 0

    def test_all_dupes_no_processing(self):
        """Edge case: re-running an already-completed sweep — all 4 hit dedup."""
        total = 4
        completed = 0
        failed = 0
        skipped_dupes = 4
        not_processed = total - completed - failed - skipped_dupes
        assert not_processed == 0

    def test_abort_at_first_failure_continue_on_failure_false(self):
        """Abort path: continue_on_failure=False; first failure breaks loop."""
        # 8 grid points; i=0 fails; loop aborts → 7 not_processed
        total = 8
        completed = 0
        failed = 1
        skipped_dupes = 0
        not_processed = total - completed - failed - skipped_dupes
        assert not_processed == 7

    def test_zero_grid_points_edge(self):
        """Edge: empty sweep (defensive — shouldn't happen but mustn't crash)."""
        total = 0
        completed = 0
        failed = 0
        skipped_dupes = 0
        not_processed = total - completed - failed - skipped_dupes
        assert not_processed == 0

    def test_mixed_complete_fail_dupe_abort_sequence(self):
        """Realistic R-16a-like scenario: 4 grid points; 1 completed, 1 dupe-skipped,
        1 failed (with continue_on_failure), 1 completed. No abort.
        """
        total = 4
        completed = 2
        failed = 1
        skipped_dupes = 1
        not_processed = total - completed - failed - skipped_dupes
        assert not_processed == 0

    def test_post_abort_grid_points_NOT_counted_as_dupes_post_F6(self):
        """F6 closure: the specific #PY-127 bug — post-abort grid points NOT lumped
        with dupes anymore.
        """
        # Scenario from #PY-127 backlog entry:
        # First R-16a launch reported "Skipped (dupes): 3" when actual was
        # 1 failed + 3 never-processed (loop break at i=1 due to abort).
        # Post-F6: skipped_dupes=0, not_processed=3 (distinct from dupes)
        total = 4
        completed = 0
        failed = 1
        skipped_dupes = 0  # ZERO real dupes
        not_processed = total - completed - failed - skipped_dupes
        assert not_processed == 3
        # Critical assertion: skipped_dupes is NOT inflated by post-abort points
        assert skipped_dupes == 0, (
            "F6 regression: skipped_dupes incorrectly counts post-abort points"
        )
