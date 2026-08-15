"""H1: ``load_manifest`` must FAIL LOUD on an unknown / typo'd stage key.

Without this guard a typo'd stage block (e.g. ``trainning:``) is silently
dropped and the real stage runs with its default ``enabled`` — the experiment
silently does the wrong thing, undetectable from the ledger.

Provenance: VALIDATION_AND_DESIGN_2026_05_30.md §12 Step 4 (H1).
"""

from pathlib import Path

import pytest

from hft_ops.manifest._field_introspection import stage_names
from hft_ops.manifest.loader import load_manifest
from hft_ops.manifest.schema import MissingGatePolicyError


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "m.yaml"
    p.write_text(body)
    return p


class TestUnknownStageRaises:
    def test_typo_stage_raises_valueerror(self, tmp_path: Path):
        m = _write(
            tmp_path,
            """
experiment:
  name: typo
  contract_version: "2.2"
pipeline_root: "."
stages:
  validation:
    enabled: true
    on_fail: warn
    min_ic: 0.05
    min_ic_count: 2
    min_return_std_bps: 5.0
    min_stability: 2.0
  trainning:
    enabled: true
""",
        )
        with pytest.raises(ValueError, match="trainning"):
            load_manifest(m)

    def test_error_lists_valid_stages(self, tmp_path: Path):
        m = _write(
            tmp_path,
            """
experiment:
  name: typo2
  contract_version: "2.2"
pipeline_root: "."
stages:
  validation:
    enabled: true
    on_fail: warn
    min_ic: 0.05
    min_ic_count: 2
    min_return_std_bps: 5.0
    min_stability: 2.0
  bogus_stage:
    enabled: true
""",
        )
        with pytest.raises(ValueError) as exc:
            load_manifest(m)
        msg = str(exc.value)
        # Surfaces the valid set so the operator can self-correct.
        assert "training" in msg and "backtesting" in msg

    def test_all_known_stages_load_clean(self, tmp_path: Path):
        # A manifest naming EVERY real stage must NOT raise (no false positive).
        body_stages = "\n".join(
            f"  {n}:\n    enabled: false" for n in sorted(stage_names())
        )
        m = _write(
            tmp_path,
            f"""
experiment:
  name: all_known
  contract_version: "2.2"
pipeline_root: "."
stages:
  validation:
    enabled: true
    on_fail: warn
    min_ic: 0.05
    min_ic_count: 2
    min_return_std_bps: 5.0
    min_stability: 2.0
{body_stages}
""",
        )
        manifest = load_manifest(m)  # must not raise
        assert manifest.experiment.name == "all_known"

    def test_no_stages_key_does_not_trip_the_UNKNOWN_STAGE_guard(self, tmp_path: Path):
        """Absent ``stages:`` is not an unknown-stage error — but it IS a
        missing-gate-policy error, and that distinction is the point.

        2026-08-15 (ruling R2): an absent ``stages:`` block leaves
        ``validation.enabled`` at its dataclass default of True, so a
        stages-less manifest silently enabled the ONE mandatory §13 gate and
        ran it on five thresholds nobody wrote down. It now fails loud. This
        test asserts the guard under test (unknown STAGE NAME) is not what
        fires — the error must name the gate policy, not a stage typo.
        """
        m = _write(
            tmp_path,
            """
experiment:
  name: no_stages
  contract_version: "2.2"
pipeline_root: "."
""",
        )
        with pytest.raises(MissingGatePolicyError) as exc:
            load_manifest(m)
        assert "Unknown stage" not in str(exc.value)
        assert "stages.validation is ENABLED" in str(exc.value)

    def test_no_stages_key_loads_once_the_gate_is_declared(self, tmp_path: Path):
        """The original intent, preserved: nothing about an otherwise-empty
        ``stages:`` block trips the unknown-stage guard."""
        m = _write(
            tmp_path,
            """
experiment:
  name: no_stages
  contract_version: "2.2"
pipeline_root: "."
stages:
  validation:
    enabled: false
""",
        )
        manifest = load_manifest(m)  # must not raise
        assert manifest.experiment.name == "no_stages"
