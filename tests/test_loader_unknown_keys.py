"""H2: every stage loader + the top-level manifest loader WARN on unknown keys.

Pre-cluster, ONLY ``_build_backtesting`` warned; the other 7 stages + the
top-level manifest silently dropped unknown keys (hft-rules §8 violation —
operator edits ``training.feature_set`` thinking it changes the run; it
silently doesn't). The known-key sets are now derived from the introspection
SSoT (``known_keys_for_stage`` / ``ExperimentManifest`` fields), retiring the
per-stage hand-copied ``_KNOWN_BACKTESTING_KEYS``.

Provenance: VALIDATION_AND_DESIGN_2026_05_30.md §12 Step 5 (H2).
"""

import warnings
from pathlib import Path

from hft_ops.manifest._field_introspection import stage_names
from hft_ops.manifest.loader import (
    _build_extraction,
    _build_training,
    load_manifest,
)


def _runtime_warns(caught):
    return [w for w in caught if w.category is RuntimeWarning]


class TestStageKeyWarnings:
    def test_training_unknown_key_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _build_training({"enabled": True, "bogus_key": 1})
        runtime = _runtime_warns(caught)
        # R2 trap #2: assert the TOTAL RuntimeWarning count (not just a
        # substring-filtered subset), so a future message-prefix change can't
        # silently zero a filter and mask the warning.
        assert len(runtime) == 1, f"expected 1 WARN, got {len(runtime)}"
        msg = str(runtime[0].message)
        assert "TrainingStage loader" in msg  # stage-attributable token
        assert "bogus_key" in msg

    def test_extraction_unknown_key_warns(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _build_extraction({"enabled": True, "not_a_field": "x"})
        runtime = _runtime_warns(caught)
        assert len(runtime) == 1
        assert "ExtractionStage loader" in str(runtime[0].message)
        assert "not_a_field" in str(runtime[0].message)

    def test_known_keys_only_no_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _build_training(
                {"enabled": True, "config": "x.yaml", "horizon_value": 10}
            )
        assert _runtime_warns(caught) == []

    def test_every_stage_warns_on_unknown_key(self, tmp_path: Path):
        """EVERY stage block warns on an unknown key (proves the guard reaches
        all 8 stages, not just backtesting)."""
        for sname in sorted(stage_names()):
            body = f"""
experiment:
  name: h2_{sname}
  contract_version: "2.2"
pipeline_root: "."
stages:
  {sname}:
    definitely_not_a_field: 1
"""
            p = tmp_path / f"m_{sname}.yaml"
            p.write_text(body)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                load_manifest(p)
            stage_warns = [
                w
                for w in _runtime_warns(caught)
                if "definitely_not_a_field" in str(w.message)
            ]
            assert len(stage_warns) == 1, (
                f"stage {sname!r}: expected 1 unknown-key WARN, got "
                f"{[str(w.message) for w in _runtime_warns(caught)]}"
            )


class TestTopLevelKeyWarning:
    def test_unknown_top_level_key_warns(self, tmp_path: Path):
        p = tmp_path / "m.yaml"
        p.write_text(
            """
experiment:
  name: toplevel
  contract_version: "2.2"
pipeline_root: "."
profiler_references:
  - some_ref
stages:
  extraction:
    enabled: false
"""
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load_manifest(p)
        toplevel = [
            w
            for w in _runtime_warns(caught)
            if "profiler_references" in str(w.message)
        ]
        assert len(toplevel) == 1, (
            f"expected 1 top-level unknown-key WARN; got "
            f"{[str(w.message) for w in _runtime_warns(caught)]}"
        )

    def test_known_top_level_keys_no_warn(self, tmp_path: Path):
        p = tmp_path / "m.yaml"
        p.write_text(
            """
experiment:
  name: clean
  contract_version: "2.2"
pipeline_root: "."
stages:
  extraction:
    enabled: false
"""
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load_manifest(p)
        assert not [
            w
            for w in _runtime_warns(caught)
            if "unknown top-level keys" in str(w.message)
        ]
