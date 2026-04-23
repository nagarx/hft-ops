"""Phase 7.5-B.3 (2026-04-23): loader WARN on unknown backtesting keys.

Closes hft-rules §8 violation surfaced by 5-agent final-validation-round:
3 production HMHP manifests declare `readability:` / `holding:` / `costs:`
sub-blocks at `backtesting:` top-level that `BacktestingStage` schema does
NOT declare. Prior loader silently dropped these (live-fs verified:
`stage.readability` was AttributeError post-load).

Post Phase 7.5-B.3: loader emits `RuntimeWarning` citing the dropped keys
+ migration guidance (use `extra_args:` list) + full known-key set.
Does NOT raise — operators continue to get functional parse; the warning
converts silent-drop to operator-visible diagnostic.

Operator-facing impact of silent-drop (pre-fix):
- `nvda_hmhp_128feat_arcx_h10.yaml` declares `costs: {exchange: ARCX}` but
  runner never sees → subprocess defaults `--exchange XNAS` → ARCX
  experiment runs with XNAS cost model → silent-wrong-result
- Coincidentally no-op today for `readability.min_agreement: 1.0` which
  matches script default 1.0, but immediately silent-breaking if operator
  sets `min_agreement: 0.8`
"""

from __future__ import annotations

import warnings

import pytest

from hft_ops.manifest.loader import _build_backtesting


class TestUnknownKeyWarning:
    """Locks the warning contract post Phase 7.5-B.3."""

    def test_unknown_key_readability_warns(self):
        """`readability:` top-level block → RuntimeWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _build_backtesting({
                "enabled": True,
                "script": "x.py",
                "signals_dir": "sigs",
                "readability": {"min_agreement": 1.0, "min_confidence": 0.65},
            })
        ours = [w for w in caught if "BacktestingStage loader" in str(w.message)]
        assert len(ours) == 1, (
            f"Expected 1 WARN on unknown key; got {len(ours)}"
        )
        msg = str(ours[0].message)
        assert "readability" in msg
        assert "extra_args" in msg  # migration hint present
        assert ours[0].category is RuntimeWarning

    def test_unknown_key_costs_warns(self):
        """`costs:` (the ARCX silent-drop culprit) → RuntimeWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _build_backtesting({
                "enabled": True,
                "script": "x.py",
                "signals_dir": "sigs",
                "costs": {"exchange": "ARCX"},
            })
        ours = [w for w in caught if "BacktestingStage loader" in str(w.message)]
        assert len(ours) == 1
        assert "costs" in str(ours[0].message)

    def test_unknown_key_holding_warns(self):
        """`holding:` top-level block → RuntimeWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _build_backtesting({
                "enabled": True,
                "script": "x.py",
                "signals_dir": "sigs",
                "holding": {"type": "horizon_aligned", "hold_events": 10},
            })
        ours = [w for w in caught if "BacktestingStage loader" in str(w.message)]
        assert len(ours) == 1
        assert "holding" in str(ours[0].message)

    def test_multiple_unknown_keys_single_warning(self):
        """3 unknown keys → ONE warning citing all 3 (not 3 separate warns)."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _build_backtesting({
                "enabled": True,
                "script": "x.py",
                "signals_dir": "sigs",
                "readability": {"min_agreement": 1.0},
                "holding": {"type": "horizon_aligned"},
                "costs": {"exchange": "ARCX"},
            })
        ours = [w for w in caught if "BacktestingStage loader" in str(w.message)]
        assert len(ours) == 1, "Expected single consolidated WARN"
        msg = str(ours[0].message)
        # All 3 dropped keys cited (sorted for determinism — matches fix)
        assert "costs" in msg
        assert "holding" in msg
        assert "readability" in msg

    def test_known_keys_only_no_warning(self):
        """Manifest using ONLY declared schema fields → no WARN."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _build_backtesting({
                "enabled": True,
                "script": "x.py",
                "signals_dir": "sigs",
                "model_checkpoint": "",
                "data_dir": "",
                "horizon_idx": 0,
                "params": {"initial_capital": 100_000.0},
                "params_file": "",
                "extra_args": ["--exchange", "ARCX"],
            })
        ours = [w for w in caught if "BacktestingStage loader" in str(w.message)]
        assert ours == [], (
            f"Known-keys-only manifest should NOT emit WARN; got "
            f"{[str(w.message) for w in ours]}"
        )

    def test_empty_manifest_no_warning(self):
        """Empty dict → no WARN (all defaults, no unknown keys)."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _build_backtesting({})
        ours = [w for w in caught if "BacktestingStage loader" in str(w.message)]
        assert ours == []

    def test_stage_construction_succeeds_despite_warn(self):
        """WARN does not BLOCK construction — stage still parses + defaults
        correctly. The operator continues to get a working stage; only
        their dropped-intent is now visible."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stage = _build_backtesting({
                "enabled": True,
                "script": "x.py",
                "signals_dir": "sigs",
                "readability": {"min_agreement": 1.0},
            })
        assert stage.enabled is True
        assert stage.script == "x.py"
        assert stage.signals_dir == "sigs"
        # Unknown keys are dropped — accessing them would AttributeError.
        # This is the INTENT of the WARN: operator sees the drop.
        assert not hasattr(stage, "readability")
