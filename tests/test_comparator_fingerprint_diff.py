"""Phase V.1 L2.2 (2026-04-21): `diff_experiments` compatibility_fingerprint
surface regression tests.

Closes Agent 4 H3 gap surfaced by Phase V post-audit cross-cutting review.
`hft-ops ledger diff A B` previously only diffed extraction_config /
training_config / backtest_params / training_metrics / backtest_metrics —
the V.A.4 compatibility_fingerprint "trust column" was surfaced in
`ledger list --compatibility-fp <hex>` (filter) but NOT in the diff view.

Locks:
  T1: matching fingerprints → `compatibility_fingerprint` field is None.
  T2: differing fingerprints → field is tuple `(fp_a, fp_b)`.
  T3: pre-V.A.4 records (both None) → field is None (match).
  T4: asymmetric (one None, one set) → field is tuple — surfaces
      provenance asymmetry to the operator.
"""

from __future__ import annotations

import pytest

from hft_contracts.experiment_record import ExperimentRecord
from hft_ops.ledger.comparator import diff_experiments


_HEX_A = "a" * 64  # valid 64-hex
_HEX_B = "b" * 64  # valid 64-hex


class TestCompatibilityFingerprintDiff:
    """L2.2 regression lock — diff surfaces fingerprint divergence."""

    def test_matching_fingerprints_yield_none(self):
        """Both records carry the same fingerprint → None."""
        a = ExperimentRecord(experiment_id="exp_a", compatibility_fingerprint=_HEX_A)
        b = ExperimentRecord(experiment_id="exp_b", compatibility_fingerprint=_HEX_A)
        result = diff_experiments(a, b)
        assert "compatibility_fingerprint" in result, (
            f"Expected 'compatibility_fingerprint' key in diff result; "
            f"got keys: {list(result.keys())}"
        )
        assert result["compatibility_fingerprint"] is None

    def test_differing_fingerprints_yield_tuple(self):
        """Different fingerprints → tuple (fp_a, fp_b)."""
        a = ExperimentRecord(experiment_id="exp_a", compatibility_fingerprint=_HEX_A)
        b = ExperimentRecord(experiment_id="exp_b", compatibility_fingerprint=_HEX_B)
        result = diff_experiments(a, b)
        assert result["compatibility_fingerprint"] == (_HEX_A, _HEX_B), (
            f"Expected tuple ('a'×64, 'b'×64); got "
            f"{result['compatibility_fingerprint']!r}"
        )

    def test_both_unset_yields_none(self):
        """Both records carry None → match → None. Pre-V.A.4 records
        both without fingerprint harvest."""
        a = ExperimentRecord(experiment_id="exp_a")  # default None
        b = ExperimentRecord(experiment_id="exp_b")  # default None
        result = diff_experiments(a, b)
        assert result["compatibility_fingerprint"] is None

    def test_asymmetric_one_set_one_unset_yields_tuple(self):
        """One record has fingerprint, other doesn't → tuple (fp, None) or
        (None, fp). Surfaces provenance asymmetry — legacy record vs
        post-V.A.4 record."""
        a = ExperimentRecord(experiment_id="exp_a", compatibility_fingerprint=_HEX_A)
        b = ExperimentRecord(experiment_id="exp_b", compatibility_fingerprint=None)
        result = diff_experiments(a, b)
        assert result["compatibility_fingerprint"] == (_HEX_A, None)
        # Reverse order (b vs a)
        result_reverse = diff_experiments(b, a)
        assert result_reverse["compatibility_fingerprint"] == (None, _HEX_A)

    def test_other_diff_fields_still_present(self):
        """Locks that the new field is ADDITIVE — config_diffs,
        metric_diffs, experiment_a/b all still populated per pre-V.1.L2.2
        contract."""
        a = ExperimentRecord(experiment_id="exp_a", compatibility_fingerprint=_HEX_A)
        b = ExperimentRecord(experiment_id="exp_b", compatibility_fingerprint=_HEX_B)
        result = diff_experiments(a, b)
        for expected_key in [
            "experiment_a",
            "experiment_b",
            "config_diffs",
            "metric_diffs",
            "compatibility_fingerprint",
        ]:
            assert expected_key in result, f"Missing key: {expected_key}"
        assert result["experiment_a"] == "exp_a"
        assert result["experiment_b"] == "exp_b"
