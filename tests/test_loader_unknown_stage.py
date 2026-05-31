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
{body_stages}
""",
        )
        manifest = load_manifest(m)  # must not raise
        assert manifest.experiment.name == "all_known"

    def test_no_stages_key_is_fine(self, tmp_path: Path):
        # Absent ``stages:`` (defaults everywhere) must not trip the guard.
        m = _write(
            tmp_path,
            """
experiment:
  name: no_stages
  contract_version: "2.2"
pipeline_root: "."
""",
        )
        manifest = load_manifest(m)  # must not raise
        assert manifest.experiment.name == "no_stages"
