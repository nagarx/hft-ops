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
from hft_contracts.provenance import Provenance
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
            # Phase Y / γ-1 LITE / #PY-95 (2026-05-10): also-additive
            "experiment_provenance_hash",
            "model_config_hash",
            # Step 5 (2026-05-31): producer_commits divergence — also additive
            "producer_commits",
        ]:
            assert expected_key in result, f"Missing key: {expected_key}"
        assert result["experiment_a"] == "exp_a"
        assert result["experiment_b"] == "exp_b"


# =============================================================================
# Phase Y / γ-1 LITE / #PY-95 (2026-05-10): diff surfaces
# experiment_provenance_hash + model_config_hash divergence
# =============================================================================


class TestExperimentProvenanceHashDiff:
    """#PY-95 closure: diff surfaces 4-source composer fingerprint divergence."""

    def test_matching_eph_yield_none(self):
        a = ExperimentRecord(experiment_id="exp_a", experiment_provenance_hash=_HEX_A)
        b = ExperimentRecord(experiment_id="exp_b", experiment_provenance_hash=_HEX_A)
        result = diff_experiments(a, b)
        assert result["experiment_provenance_hash"] is None

    def test_differing_eph_yield_tuple(self):
        a = ExperimentRecord(experiment_id="exp_a", experiment_provenance_hash=_HEX_A)
        b = ExperimentRecord(experiment_id="exp_b", experiment_provenance_hash=_HEX_B)
        result = diff_experiments(a, b)
        assert result["experiment_provenance_hash"] == (_HEX_A, _HEX_B)

    def test_both_eph_unset_yields_none(self):
        """Both records carry None → match. Pre-Phase-Y records both
        missing harvest."""
        a = ExperimentRecord(experiment_id="exp_a")
        b = ExperimentRecord(experiment_id="exp_b")
        result = diff_experiments(a, b)
        assert result["experiment_provenance_hash"] is None

    def test_asymmetric_eph_yields_tuple(self):
        """Legacy vs post-Phase-Y record → asymmetric divergence surfaced."""
        a = ExperimentRecord(experiment_id="exp_a", experiment_provenance_hash=_HEX_A)
        b = ExperimentRecord(experiment_id="exp_b")
        result = diff_experiments(a, b)
        assert result["experiment_provenance_hash"] == (_HEX_A, None)


class TestModelConfigHashDiff:
    """#PY-95 closure: diff surfaces model-axis identity divergence."""

    def test_matching_mch_yields_none(self):
        a = ExperimentRecord(
            experiment_id="exp_a",
            training_config={"model_config_hash": _HEX_A},
        )
        b = ExperimentRecord(
            experiment_id="exp_b",
            training_config={"model_config_hash": _HEX_A},
        )
        result = diff_experiments(a, b)
        assert result["model_config_hash"] is None

    def test_differing_mch_yields_tuple(self):
        """Same data, different arch → mch differs → tuple surfaced.

        Use case: cross-architecture-on-same-data ablation
        (TLOB vs TemporalRidge on identical signals).
        """
        a = ExperimentRecord(
            experiment_id="exp_a",
            training_config={"model_config_hash": _HEX_A},
        )
        b = ExperimentRecord(
            experiment_id="exp_b",
            training_config={"model_config_hash": _HEX_B},
        )
        result = diff_experiments(a, b)
        assert result["model_config_hash"] == (_HEX_A, _HEX_B)

    def test_both_mch_missing_yields_none(self):
        """training_config absent or lacking key on both → None."""
        a = ExperimentRecord(experiment_id="exp_a", training_config={})
        b = ExperimentRecord(experiment_id="exp_b", training_config={})
        result = diff_experiments(a, b)
        assert result["model_config_hash"] is None

    def test_asymmetric_mch_yields_tuple(self):
        """Legacy record (no mch) vs post-Phase-Y record (mch populated)."""
        a = ExperimentRecord(
            experiment_id="exp_a",
            training_config={"model_config_hash": _HEX_A},
        )
        b = ExperimentRecord(experiment_id="exp_b", training_config={})
        result = diff_experiments(a, b)
        assert result["model_config_hash"] == (_HEX_A, None)

    def test_unrelated_training_config_keys_ignored(self):
        """Locks the mch extraction is independent of other training_config
        keys — same mch = no divergence even with other diffs."""
        a = ExperimentRecord(
            experiment_id="exp_a",
            training_config={"model_config_hash": _HEX_A, "lr": 0.001},
        )
        b = ExperimentRecord(
            experiment_id="exp_b",
            training_config={"model_config_hash": _HEX_A, "lr": 0.002},
        )
        result = diff_experiments(a, b)
        # mch matches; surfaced via None
        assert result["model_config_hash"] is None
        # But config_diffs DOES surface lr divergence (existing surface)
        assert any(
            "lr" in str(d[0]).lower() for d in result["config_diffs"]
        ), "config_diffs should surface lr divergence even when mch matches"


# =============================================================================
# Step 5 (2026-05-31): diff surfaces producer_commits (Foundation-Integrity
# producer-code lineage) divergence. The phase CAPTURES it (extraction.py:141,
# 200) but diff never read record.provenance — so two records built from
# DIFFERENT reconstructor/extractor commits showed IDENTICAL diff output,
# defeating the phase's purpose (catch silently-wrong Rust-producer lineage).
# =============================================================================
class TestProducerCommitsDiff:
    @staticmethod
    def _rec(eid, producer_commits):
        return ExperimentRecord(
            experiment_id=eid,
            provenance=Provenance(producer_commits=producer_commits),
        )

    def test_matching_producer_commits_yield_none(self):
        pc = {"extractor_git_sha": "a" * 40, "reconstructor_git_sha": "b" * 40,
              "completeness": "full"}
        result = diff_experiments(self._rec("a", pc), self._rec("b", dict(pc)))
        assert result["producer_commits"] is None

    def test_differing_reconstructor_sha_yields_tuple(self):
        """Same extractor, DIFFERENT reconstructor commit -> the exact silent
        lineage drift the phase exists to catch."""
        pc_a = {"extractor_git_sha": "a" * 40, "reconstructor_git_sha": "b" * 40,
                "completeness": "full"}
        pc_b = {"extractor_git_sha": "a" * 40, "reconstructor_git_sha": "c" * 40,
                "completeness": "full"}
        result = diff_experiments(self._rec("a", pc_a), self._rec("b", pc_b))
        assert result["producer_commits"] == (pc_a, pc_b)

    def test_both_empty_yield_none(self):
        result = diff_experiments(self._rec("a", {}), self._rec("b", {}))
        assert result["producer_commits"] is None

    def test_asymmetric_one_populated_one_empty_yields_tuple(self):
        """Post-P1a record (captured) vs a record that ran a path with no
        capture (e.g. the deferred skip_if_exists) -> surfaced asymmetry."""
        pc_a = {"extractor_git_sha": "a" * 40, "completeness": "partial"}
        result = diff_experiments(self._rec("a", pc_a), self._rec("b", {}))
        assert result["producer_commits"] == (pc_a, {})

    def test_default_provenance_records_yield_none(self):
        """Records built WITHOUT an explicit provenance (default Provenance has
        empty producer_commits) -> both empty -> None (no false divergence)."""
        a = ExperimentRecord(experiment_id="a")
        b = ExperimentRecord(experiment_id="b")
        result = diff_experiments(a, b)
        assert result["producer_commits"] is None
