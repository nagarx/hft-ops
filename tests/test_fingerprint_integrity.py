"""Phase R-17 F2: regression tests for #PY-130 + #PY-136 fingerprint integrity.

Tests dedup.py `_extract_fingerprint_fields` + nested `_strip` function for:

1. **Baseline set-equality** to the production SSoT (13 exclude_keys; any drift
   REQUIRES this test update — H4 §12 Step 6 retired the stale len==12 lock)
2. **Synthetic collapse hazard** (#PY-130) — dormant in production, REAL bug
3. **List-of-dicts sister-site leak** (#PY-136 / N2-2) — dormant in production
4. **Production manifest distinctness** — locks empirical 2026-05-10/11 results

References:
- Phase R-17 v2 design: POST_R16A_DESIGN_PHASE_2026_05_11.md §10
- PHASE_P_BACKLOG.md #PY-130 + #PY-136
- Anti-drift mandate (CLAUDE.md banner #1): F2 is REGRESSION TEST ONLY, not
  sentinel-fix. The sentinel-fix is DEFERRED to a future MAJOR schema bump
  cycle to avoid rotating all prior fingerprints (R-16a + cycle5 + #PY-94 closures).
  Tests in TestStripCollapseHazardDocumented + TestStripListOfDictsLeakDocumented
  assert CURRENT (buggy) behavior; the sentinel-fix would INVERT these assertions
  (that's how we know it landed).

Per G4 adversarial verdict (2026-05-11 late-night): F2c MUST be NON-xfail —
asserts the synthetic crafted collapse case IS reproducible TODAY (proves
hazard exists in some configurations, even if dormant in current production).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from hft_ops.ledger.dedup import (
    _FINGERPRINT_EXCLUDE_KEYS,
    _extract_fingerprint_fields,
    compute_fingerprint,
)
from hft_ops.manifest.loader import load_manifest
from hft_ops.paths import PipelinePaths


# Pipeline root: 4-up from this test file's location (tests/test_*.py → tests/ → hft-ops/ → HFT-pipeline-v2/)
PIPELINE_ROOT = Path(__file__).resolve().parent.parent.parent


# =============================================================================
# TestExcludeKeysFrozenBaseline — STRICT 12-key membership lock
# =============================================================================


class TestExcludeKeysFrozenBaseline:
    """Phase R-17 F2 + H4 (§12 Step 6): lock the 13 exclude_keys via set-equality
    to the production SSoT ``dedup._FINGERPRINT_EXCLUDE_KEYS``.

    Strict membership policy per G4 verdict (option a — exact verbatim assertion):
    adding or removing any key REQUIRES updating ``EXPECTED_EXCLUDE_KEYS`` here
    AND adding a sister regression test for the new key's behavior.

    Rationale: ``exclude_keys`` IS the SSoT for "treatment vs observation"
    semantics. Silent additions could create new collapse hazards (the #PY-130
    class). Strictness forces explicit acknowledgement of the contract change.
    """

    EXPECTED_EXCLUDE_KEYS = frozenset({
        # Metadata (Phase 3 baseline — no semantic effect on training outcome)
        "name", "description", "tags", "version",
        # Output paths + logging (Phase 3 baseline)
        "output_dir", "log_level", "verbose",
        # Experiment block (Phase 5 FULL-A — manifest organizational metadata)
        "experiment",
        # Phase 8C-α (2026-04-20): post-training permutation importance is OBSERVATION
        "importance",
        # Phase 8C-α (2026-04-20): post-training artifacts are OBSERVATION on output side
        "artifacts",
        # Phase DESIGN-1 A.2 (2026-05-10): RNG state in checkpoint — per-call OBSERVATION
        "rng_state",
        # Phase DESIGN-1 G-1 (2026-05-10): callback state in checkpoint — per-call OBSERVATION
        "callback_state",
        # P1a (2026-05-30): producer_commits on Provenance — which-code-built-it OBSERVATION
        "producer_commits",
    })

    def test_baseline_set_equals_production_ssot(self):
        """The hand-mirrored baseline MUST equal the production SSoT BY VALUE.

        H4 (VALIDATION_AND_DESIGN_2026_05_30.md §12 Step 6): the prior test
        asserted only ``len == 12`` against its OWN hand-copied set, which had
        silently diverged from production (12 vs 13 — missing
        ``producer_commits``, added 2026-05-30). Comparing to the imported
        production object ``_FINGERPRINT_EXCLUDE_KEYS`` forces any future change
        to the exclude set to be acknowledged HERE and paired with a sister
        behavioral test (the parametrized strip tests below auto-cover every key
        in this set).
        """
        assert self.EXPECTED_EXCLUDE_KEYS == _FINGERPRINT_EXCLUDE_KEYS, (
            "Baseline drift between this test's EXPECTED_EXCLUDE_KEYS and the "
            "production SSoT dedup._FINGERPRINT_EXCLUDE_KEYS. "
            f"In production only: "
            f"{sorted(_FINGERPRINT_EXCLUDE_KEYS - self.EXPECTED_EXCLUDE_KEYS)}. "
            f"In test only: "
            f"{sorted(self.EXPECTED_EXCLUDE_KEYS - _FINGERPRINT_EXCLUDE_KEYS)}. "
            "If you intentionally changed the exclude set, update "
            "EXPECTED_EXCLUDE_KEYS AND add a sister regression test."
        )

    @pytest.mark.parametrize("excluded_key", sorted(EXPECTED_EXCLUDE_KEYS))
    def test_each_baseline_key_stripped_at_top_level(self, excluded_key: str):
        """Each baseline exclude_key MUST strip from top-level position."""
        cfg = {excluded_key: "should_be_stripped", "preserved_field": {"keep": "yes"}}
        result = _extract_fingerprint_fields(cfg)
        assert excluded_key not in result, (
            f"Baseline exclude_key '{excluded_key}' was preserved at top-level (REGRESSION). "
            f"_extract_fingerprint_fields no longer strips this key."
        )
        assert result["preserved_field"]["keep"] == "yes"

    @pytest.mark.parametrize("excluded_key", sorted(EXPECTED_EXCLUDE_KEYS))
    def test_each_baseline_key_stripped_at_nested_level(self, excluded_key: str):
        """Each baseline exclude_key MUST strip from nested positions inside other dicts."""
        cfg = {
            "data": {excluded_key: "should_be_stripped", "feature_count": 98},
            "model": {"hidden_size": 64, excluded_key: "also_stripped"},
        }
        result = _extract_fingerprint_fields(cfg)
        assert excluded_key not in result["data"], (
            f"exclude_key '{excluded_key}' was preserved at data.* nested level"
        )
        assert excluded_key not in result["model"], (
            f"exclude_key '{excluded_key}' was preserved at model.* nested level"
        )
        # Treatment-tier fields still preserved
        assert result["data"]["feature_count"] == 98
        assert result["model"]["hidden_size"] == 64


# =============================================================================
# TestStripCollapseHazardDocumented — #PY-130 dormant bug regression target
# =============================================================================


class TestStripCollapseHazardDocumented:
    """Phase R-17 F2c: #PY-130 latent fingerprint integrity hazard.

    The ``_strip`` function at dedup.py:701-712 drops dict subtrees whose
    children ALL appear in ``exclude_keys`` (line 708 ``if stripped:`` evaluates
    False on empty dict, so the subtree is removed entirely instead of preserved
    as ``{}``). Two configs differing ONLY in such a subtree produce IDENTICAL
    post-strip outputs → IDENTICAL fingerprints → ledger conflation.

    **Status**: DORMANT in current production. The 12 exclude_keys
    (name + description + tags + version + output_dir + log_level + verbose +
    experiment + importance + artifacts + rng_state + callback_state) do not
    occur as "all children" of any sub-dict in current production manifests
    (verified empirically: cycle5_multi_arm 2026-05-10 produced 12 distinct
    fingerprints; cycle6_r16a 2026-05-11 produced 4 distinct).

    **Activation hazard**: any future schema addition where a new exclude_key
    creates an all-excluded subtree silently activates the bug.

    **Sentinel-fix (DEFERRED to Phase R-19/R-20 with MAJOR schema bump)**:
    would change line 708 to ``result[k] = stripped`` (always preserve subtree,
    even if empty post-strip). This test ASSERTS CURRENT (buggy) behavior —
    when the sentinel-fix lands, this test breaks (that's the signal).

    Per G4 verdict (2026-05-11): NON-xfail. Test PASSES today (proving the
    hazard is reproducible on synthetic case); flips to FAIL when sentinel-fix
    lands (signals deferred work completed).
    """

    def test_synthetic_all_excluded_subtree_collapses(self):
        """#PY-130 reproducer: two configs differing ONLY in an all-exclude_keys
        subtree produce IDENTICAL post-strip outputs.

        Config A and Config B differ in two ways:
          - checkpoints.output_dir: "path/A" vs "path/B"
          - checkpoints.rng_state: differs in nested dict

        Both fields are in exclude_keys. After _strip:
          - checkpoints subtree becomes {} (both keys excluded)
          - line 708 `if stripped:` evaluates False → drops checkpoints entirely
          - Result: both configs strip to {"data": {"feature_count": 98}}
        """
        config_a: Dict[str, Any] = {
            "data": {"feature_count": 98},
            "checkpoints": {
                "output_dir": "path/A",
                "rng_state": {"torch": "stateA"},
            },
        }
        config_b: Dict[str, Any] = {
            "data": {"feature_count": 98},
            "checkpoints": {
                "output_dir": "path/B",
                "rng_state": {"torch": "stateB"},
            },
        }
        result_a = _extract_fingerprint_fields(config_a)
        result_b = _extract_fingerprint_fields(config_b)

        # HAZARD ASSERTION: both produce identical output despite differing inputs
        assert result_a == result_b, (
            "#PY-130 sentinel-fix appears to have landed (result_a != result_b). "
            "If sentinel-fix was intentional, update this test to assert "
            "preservation of empty subtree instead of collapse."
        )
        # The hazard manifests as the checkpoints subtree being dropped entirely
        assert "checkpoints" not in result_a, (
            "#PY-130 sentinel-fix appears to have landed (checkpoints subtree preserved). "
            "Update this test to assert result_a['checkpoints'] == {} instead."
        )

    def test_synthetic_partially_excluded_subtree_preserves_non_excluded(self):
        """Sanity check: a subtree with MIXED excluded + non-excluded keys is preserved.

        Confirms _strip's recursion correctly handles the MIXED case (only the
        collapse-to-empty path is buggy; partial-strip preserves the subtree
        with non-excluded fields).
        """
        cfg: Dict[str, Any] = {
            "checkpoints": {
                "output_dir": "should_strip",  # in exclude_keys
                "ckpt_format": "preserved",  # NOT in exclude_keys
            },
        }
        result = _extract_fingerprint_fields(cfg)
        assert "output_dir" not in result["checkpoints"]
        assert result["checkpoints"]["ckpt_format"] == "preserved"


# =============================================================================
# TestStripListOfDictsLeakDocumented — #PY-136 / N2-2 sister-site bug
# =============================================================================


class TestStripListOfDictsLeakDocumented:
    """Phase R-17 F2 add-on: #PY-136 sister-site fingerprint integrity hazard.

    The ``_strip`` function recurses into dict values via ``isinstance(v, dict)``
    (line 706). LIST values pass through verbatim at line 711
    ``else: result[k] = v``. exclude_keys nested inside list-of-dicts are NEVER
    stripped → leak into fingerprint inputs.

    **Status**: DORMANT in current production. No production manifest has a
    list-of-dicts structure containing exclude_keys at fingerprint-input paths
    (verified by Step 0 audit 2026-05-11).

    **Activation hazard**: any future schema addition where exclude_keys-containing
    dicts are stored in a list silently activates the leak.

    **Sentinel-fix (DEFERRED)**: would add ``isinstance(v, list)`` branch to
    _strip that recurses into list elements (treating each as a strip target).

    Per H2 agent recommendation: NON-xfail. Test asserts CURRENT (leaky) behavior.
    """

    def test_list_of_dicts_passes_excluded_keys_verbatim(self):
        """#PY-136 reproducer: list-of-dicts containing exclude_keys passes through unchanged.

        Demonstrates that _strip's `isinstance(v, dict)` branch (line 706) does
        NOT recurse into list values — any exclude_keys nested inside leak
        verbatim into fingerprint inputs.
        """
        cfg: Dict[str, Any] = {
            "callbacks": [
                {"callback_type": "EarlyStopping", "rng_state": "stateA"},
                {"callback_type": "ModelCheckpoint", "callback_state": "stateB"},
            ],
        }
        result = _extract_fingerprint_fields(cfg)

        # HAZARD ASSERTION: exclude_keys inside list-of-dicts leak through
        assert "callbacks" in result, "callbacks key should be preserved (not in exclude_keys)"
        assert len(result["callbacks"]) == 2

        # Each list element still contains its rng_state / callback_state (the leak)
        assert "rng_state" in result["callbacks"][0], (
            "#PY-136 sentinel-fix appears to have landed. Update this test to "
            "assert rng_state NOT in result['callbacks'][0]."
        )
        assert "callback_state" in result["callbacks"][1], (
            "#PY-136 sentinel-fix appears to have landed. Update this test to "
            "assert callback_state NOT in result['callbacks'][1]."
        )

        # Treatment-tier fields preserved
        assert result["callbacks"][0]["callback_type"] == "EarlyStopping"
        assert result["callbacks"][1]["callback_type"] == "ModelCheckpoint"

    def test_list_of_dicts_with_mixed_excluded_and_preserved(self):
        """Confirm the leak applies uniformly even when list elements differ in
        which exclude_keys they contain."""
        cfg: Dict[str, Any] = {
            "items": [
                {"id": 1, "rng_state": "x"},
                {"id": 2},  # no exclude_keys here
                {"id": 3, "callback_state": "y", "rng_state": "z"},  # two exclude_keys
            ],
        }
        result = _extract_fingerprint_fields(cfg)
        assert len(result["items"]) == 3
        # All three list elements preserve their full contents (the leak)
        assert "rng_state" in result["items"][0]
        assert "rng_state" not in result["items"][1]  # never had it
        assert "callback_state" in result["items"][2]
        assert "rng_state" in result["items"][2]


# =============================================================================
# TestProductionManifestFingerprintIntegrity — Hybrid (3 baseline + glob N-1)
# =============================================================================


class TestProductionManifestFingerprintIntegrity:
    """Phase R-17 F2: lock production-manifest fingerprint integrity.

    Per G4 hybrid recommendation (option c): test 3 specific baseline manifests
    that empirically produced distinct fingerprints in cycle5_multi_arm 2026-05-10
    + cycle6_r16a 2026-05-11, PLUS a glob N-1 invariant catching future regressions.

    These tests load REAL production manifests via _extract_fingerprint_fields on
    the raw YAML dict (NOT full compute_fingerprint which requires resolved trainer
    config paths). This isolates the _strip behavior from manifest-loading concerns.
    """

    BASELINE_MANIFESTS = [
        "hft-ops/experiments/sweeps/cycle5_multi_arm.yaml",
        "hft-ops/experiments/sweeps/cycle6_r16a_point_vs_peak_H60.yaml",
        "hft-ops/experiments/nvda_hmhp_40feat_xnas_h10.yaml",
    ]

    def _load_raw_manifest(self, rel_path: str) -> Dict[str, Any]:
        """Load manifest YAML as raw dict (no schema validation)."""
        abs_path = PIPELINE_ROOT / rel_path
        if not abs_path.exists():
            pytest.skip(f"Production manifest not found: {abs_path}")
        with abs_path.open() as f:
            data = yaml.safe_load(f)
        if data is None:
            pytest.skip(f"Empty manifest: {abs_path}")
        return data

    def test_three_baseline_manifests_have_distinct_strip_outputs(self):
        """Each baseline manifest produces a UNIQUE post-_extract_fingerprint_fields output.

        Locks the empirical 2026-05-10/11 result: these 3 manifests must remain
        distinguishable post-strip. If Phase R-17 changes (or any future refactor)
        causes them to collapse, the bug is immediate.

        Uses canonical-JSON SHA-256 (matches dedup.py's actual hashing
        machinery via hft_contracts.canonical_hash SSoT).
        """
        from hft_contracts.canonical_hash import canonical_json_blob, sha256_hex

        fingerprints: Dict[str, str] = {}
        for rel_path in self.BASELINE_MANIFESTS:
            raw = self._load_raw_manifest(rel_path)
            stripped = _extract_fingerprint_fields(raw)
            canonical_bytes = canonical_json_blob(stripped)
            fingerprints[rel_path] = sha256_hex(canonical_bytes)

        assert len(set(fingerprints.values())) == len(self.BASELINE_MANIFESTS), (
            f"Baseline manifest fingerprint collision (Phase R-17 regression): "
            f"{fingerprints}"
        )

    def test_no_production_manifest_yields_empty_post_strip_dict(self):
        """Glob invariant: NO production manifest at hft-ops/experiments/**/*.yaml
        produces empty post-strip output.

        Empty post-strip would indicate the #PY-130 collapse hazard activating
        in real production. Currently the hazard is DORMANT — this test catches
        future regression where it activates.

        Behavioral note: this test SHOULD FAIL the moment any manifest acquires
        a structure that produces empty post-strip (e.g., a manifest containing
        ONLY exclude_keys at top-level, or all subtrees collapse).
        """
        manifests_dir = PIPELINE_ROOT / "hft-ops" / "experiments"
        if not manifests_dir.exists():
            pytest.skip(f"Experiments dir not found: {manifests_dir}")

        yaml_files = list(manifests_dir.rglob("*.yaml"))
        assert len(yaml_files) > 0, "No production manifests found under hft-ops/experiments/"

        empty_manifests = []
        for path in yaml_files:
            with path.open() as f:
                raw = yaml.safe_load(f)
            if raw is None or not isinstance(raw, dict):
                continue  # empty YAML or non-dict (e.g., list at top-level) — skip
            stripped = _extract_fingerprint_fields(raw)
            if not stripped:
                empty_manifests.append(path.relative_to(PIPELINE_ROOT))

        assert not empty_manifests, (
            f"Production manifests produced empty post-strip dicts (#PY-130 hazard ACTIVE): "
            f"{empty_manifests}. The collapse-to-empty case has activated in production. "
            f"Sentinel-fix must ship immediately."
        )

    def test_baseline_manifests_resolved_dicts_contain_treatment_axis_fields(self):
        """Sanity check: each baseline manifest's post-strip dict contains at least
        ONE treatment-axis field (i.e., not entirely metadata that got stripped).

        Failing this test would indicate either (a) the baseline manifest is
        truly all-metadata (incorrect baseline choice), or (b) _strip is over-
        stripping (regression).
        """
        for rel_path in self.BASELINE_MANIFESTS:
            raw = self._load_raw_manifest(rel_path)
            stripped = _extract_fingerprint_fields(raw)
            assert stripped, (
                f"Manifest {rel_path} produced empty post-strip dict — "
                f"either baseline is all-metadata, or _strip is over-stripping."
            )
            # At least one non-metadata top-level key must remain
            non_metadata_keys = {
                k for k in stripped
                if k not in {"name", "description", "tags", "version", "output_dir",
                             "log_level", "verbose", "experiment", "importance",
                             "artifacts", "rng_state", "callback_state"}
            }
            assert non_metadata_keys, (
                f"Manifest {rel_path} stripped to ONLY metadata keys: {set(stripped.keys())}"
            )
