"""Regression tests for 5 CRITICAL bugs surfaced in the pre-Phase-2b audit.

Each test targets a specific bug:

- C1: ``_hft_ops_compat.py`` banner was malformed by adjacent string-literal
  concatenation + ``*`` precedence (would print ``\\n=\\n=\\n=`` instead of
  ``\\n==...==\\n``).
- C2: ``_materialize_inline_config`` wrote inline trainer_configs with
  relative ``_base:`` paths that resolve under the wrong directory when the
  temp config is placed outside ``<trainer>/configs/``.
- C3: ``validator._validate_manifest`` "enabled_stages" check omitted
  ``validation`` and ``signal_export`` — single-gate or export-only manifests
  falsely raised "No stages enabled".
- C4: ``cli.ledger backfill`` exited 1 on benign fingerprint duplicate,
  breaking idempotent re-runs of ``generate_retroactive_manifests.py``.
- C5: ``DeprecationWarning`` filter hid the programmatic warning signal;
  fixed by switching to ``UserWarning`` (always default-visible).

These tests are standalone and do not depend on tmp_pipeline fixtures other
than what is documented inline.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest


# -----------------------------------------------------------------------------
# C1 & C5: _hft_ops_compat banner + warning visibility
# -----------------------------------------------------------------------------


def _load_compat_module():
    """Load the trainer-side compat helper via ``spec_from_file_location``.

    Uses the SSoT ``require_monorepo_root`` helper (Phase V.A.0) to resolve
    the monorepo layout, then loads the compat module directly via
    ``importlib.util.spec_from_file_location`` — matches the precedent set
    by ``hft_ops.ledger.dedup._load_trainer_merge_module``.

    V.A.0 audit C3 fix: the prior implementation did
    ``sys.path.insert(0, str(trainer_scripts))`` + ``importlib.import_module``
    but NEVER popped the inserted path. sys.path is process-global; any
    subsequent test importing a module name that happened to exist under
    ``lob-model-trainer/scripts/`` (e.g., ``train.py``, ``export_signals.py``)
    would silently misresolve to the trainer script. Switching to
    ``spec_from_file_location`` avoids sys.path mutation entirely —
    clean test isolation.

    Skips cleanly on standalone-clone environments (e.g., fresh CI
    without the lob-model-trainer sibling) rather than raising
    ``ModuleNotFoundError`` at import time.
    """
    from hft_contracts._testing import require_monorepo_root
    import importlib.util

    monorepo_root = require_monorepo_root(
        "lob-model-trainer/scripts/_hft_ops_compat.py",
    )
    compat_path = (
        monorepo_root / "lob-model-trainer" / "scripts" / "_hft_ops_compat.py"
    )

    # Drop any previously-imported version so the module is re-exec'd on
    # each test (idempotent load across test order permutations).
    sys.modules.pop("_hft_ops_compat", None)

    spec = importlib.util.spec_from_file_location(
        "_hft_ops_compat", compat_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Failed to load spec from {compat_path} — file exists "
            f"(require_monorepo_root verified) but cannot be loaded as "
            f"a Python module. Likely indicates a syntax error in the "
            f"compat module; run `python {compat_path}` to diagnose."
        )
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec so relative imports within
    # the module resolve correctly (standard spec-from-file-location
    # idiom per the Python 3.12 importlib docs).
    sys.modules["_hft_ops_compat"] = module
    spec.loader.exec_module(module)
    return module


class TestCompatBanner:
    """C1 + C5 regression tests."""

    def test_banner_is_wellformed_when_not_orchestrated(
        self,
        capsys,
        monkeypatch,
    ):
        """Banner must print a proper rule line, not alternating '\\n='."""
        monkeypatch.delenv("HFT_OPS_ORCHESTRATED", raising=False)
        compat = _load_compat_module()

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            compat.warn_if_not_orchestrated("unit_test_script.py")

        captured = capsys.readouterr()
        banner = captured.err

        # The rule line is 72 consecutive '=' characters. If the old bug
        # regressed, we would see interleaved newlines and '=' instead.
        assert "=" * 72 in banner, (
            "Banner should contain a 72-char '=' rule line; got "
            f"(first 200 chars):\n{banner[:200]!r}"
        )
        assert "=\n=" not in banner, (
            "Banner contains alternating '=\\n=' — adjacent string literal "
            "bug has regressed. Use explicit join() or concatenation with '+'."
        )
        assert "unit_test_script.py" in banner
        assert "hft-ops run" in banner

    def test_warning_is_userwarning_and_visible(
        self,
        capsys,
        monkeypatch,
    ):
        """DeprecationWarning is filtered by default; UserWarning is visible."""
        monkeypatch.delenv("HFT_OPS_ORCHESTRATED", raising=False)
        compat = _load_compat_module()

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            compat.warn_if_not_orchestrated("unit_test_script.py")

        # Must emit exactly one warning, of type UserWarning (C5 fix).
        assert len(record) == 1, (
            f"Expected 1 warning, got {len(record)}: "
            f"{[str(w.message) for w in record]}"
        )
        assert issubclass(record[0].category, UserWarning), (
            f"Warning should be UserWarning (default-visible), got "
            f"{record[0].category.__name__}. DeprecationWarning is filtered "
            f"when triggered from imported modules — regression from C5 fix."
        )

    def test_orchestrated_suppresses_warning(
        self,
        capsys,
        monkeypatch,
    ):
        """When HFT_OPS_ORCHESTRATED=1, no banner or warning is emitted."""
        monkeypatch.setenv("HFT_OPS_ORCHESTRATED", "1")
        compat = _load_compat_module()

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            compat.warn_if_not_orchestrated("unit_test_script.py")

        captured = capsys.readouterr()
        assert captured.err == ""
        assert len(record) == 0


# -----------------------------------------------------------------------------
# C2: inline _base: path absolutization
# -----------------------------------------------------------------------------


class TestInlineBaseAbsolutization:
    """C2 regression tests."""

    def test_absolutize_top_level_string(self, tmp_path):
        """Relative ``_base: <str>`` is rewritten to an absolute path."""
        from hft_ops.stages.training import _absolutize_inline_base_paths

        configs_root = tmp_path / "configs"
        configs_root.mkdir()

        cfg = {
            "_base": "bases/models/tlob.yaml",
            "train": {"lr": 1e-4},
        }
        _absolutize_inline_base_paths(cfg, configs_root)

        assert Path(cfg["_base"]).is_absolute()
        assert cfg["_base"].endswith("bases/models/tlob.yaml")
        assert cfg["train"] == {"lr": 1e-4}

    def test_absolutize_top_level_list(self, tmp_path):
        """Relative ``_base: [list]`` entries all rewritten (Phase 3 compat)."""
        from hft_ops.stages.training import _absolutize_inline_base_paths

        configs_root = tmp_path / "configs"
        configs_root.mkdir()

        cfg = {
            "_base": [
                "bases/models/tlob.yaml",
                "bases/datasets/nvda.yaml",
            ],
            "train": {"lr": 1e-4},
        }
        _absolutize_inline_base_paths(cfg, configs_root)

        assert isinstance(cfg["_base"], list)
        for entry in cfg["_base"]:
            assert Path(entry).is_absolute()

    def test_absolute_path_untouched(self, tmp_path):
        """Absolute ``_base:`` values are left as-is."""
        from hft_ops.stages.training import _absolutize_inline_base_paths

        configs_root = tmp_path / "configs"
        configs_root.mkdir()
        abs_path = "/etc/some/absolute/path.yaml"
        cfg = {"_base": abs_path}

        _absolutize_inline_base_paths(cfg, configs_root)

        assert cfg["_base"] == abs_path

    def test_no_base_is_noop(self, tmp_path):
        """Configs without ``_base:`` are left unchanged."""
        from hft_ops.stages.training import _absolutize_inline_base_paths

        configs_root = tmp_path / "configs"
        configs_root.mkdir()
        cfg = {"train": {"lr": 1e-4}, "data": {"path": "../some/rel/path"}}
        original = dict(cfg)

        _absolutize_inline_base_paths(cfg, configs_root)

        assert cfg == original

    def test_materialize_inline_config_absolutizes(self, tmp_path):
        """End-to-end: materialized temp YAML contains absolute _base path."""
        import yaml

        from hft_ops.stages.training import _materialize_inline_config

        configs_root = tmp_path / "trainer" / "configs"
        configs_root.mkdir(parents=True)
        # Place a fake base so the abs path resolves to something real
        base_dir = configs_root / "bases" / "models"
        base_dir.mkdir(parents=True)
        (base_dir / "tlob.yaml").write_text("model:\n  dummy: true\n")

        inline_cfg = {
            "_base": "bases/models/tlob.yaml",
            "train": {"lr": 1e-4},
        }
        out_path = tmp_path / "runs" / "exp_a" / "resolved.yaml"

        result_path = _materialize_inline_config(
            inline_cfg,
            {},
            out_path,
            trainer_configs_root=configs_root,
        )

        assert result_path == out_path
        assert out_path.exists()

        with open(out_path) as f:
            written = yaml.safe_load(f)

        assert "_base" in written
        assert Path(written["_base"]).is_absolute()
        assert Path(written["_base"]).exists()

    def test_materialize_without_configs_root_leaves_base_verbatim(
        self,
        tmp_path,
    ):
        """Backward-compat: None ``trainer_configs_root`` → no rewrite."""
        import yaml

        from hft_ops.stages.training import _materialize_inline_config

        inline_cfg = {
            "_base": "bases/models/tlob.yaml",
            "train": {"lr": 1e-4},
        }
        out_path = tmp_path / "runs" / "exp_a" / "resolved.yaml"

        _materialize_inline_config(
            inline_cfg,
            {},
            out_path,
            trainer_configs_root=None,
        )

        with open(out_path) as f:
            written = yaml.safe_load(f)

        assert written["_base"] == "bases/models/tlob.yaml"


# -----------------------------------------------------------------------------
# C3: validator enabled_stages includes validation + signal_export
# -----------------------------------------------------------------------------


class TestValidatorEnabledStages:
    """C3 regression tests."""

    def test_validation_only_manifest_not_flagged_empty(self, tmp_pipeline):
        """A manifest with ONLY validation.enabled should not warn 'no stages'."""
        import yaml

        from hft_ops.manifest.loader import load_manifest
        from hft_ops.manifest.validator import validate_manifest
        from hft_ops.paths import PipelinePaths

        manifest_dict = {
            "experiment": {"name": "validation_only", "contract_version": "2.2"},
            "pipeline_root": "..",
            "stages": {
                "extraction": {"enabled": False},
                "dataset_analysis": {"enabled": False},
                "validation": {
                    "enabled": True,
                    "on_fail": "warn",
                    "target_horizon": "10",
                    "min_ic": 0.05,
                },
                "training": {"enabled": False},
                "backtesting": {"enabled": False},
            },
        }
        manifest_path = tmp_pipeline / "hft-ops" / "experiments" / "val_only.yaml"
        with open(manifest_path, "w") as f:
            yaml.dump(manifest_dict, f)

        manifest = load_manifest(manifest_path)
        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        result = validate_manifest(manifest, paths)

        no_stages_msgs = [
            str(w) for w in result.warnings if "No stages enabled" in str(w)
        ]
        assert not no_stages_msgs, (
            "Validation-only manifest falsely flagged as empty: "
            f"{no_stages_msgs}"
        )

    def test_signal_export_only_manifest_not_flagged_empty(self, tmp_pipeline):
        """signal_export-only manifest should not trigger 'no stages' warning."""
        import yaml

        from hft_ops.manifest.loader import load_manifest
        from hft_ops.manifest.validator import validate_manifest
        from hft_ops.paths import PipelinePaths

        manifest_dict = {
            "experiment": {"name": "export_only", "contract_version": "2.2"},
            "pipeline_root": "..",
            "stages": {
                "extraction": {"enabled": False},
                "dataset_analysis": {"enabled": False},
                "validation": {"enabled": False},
                "training": {"enabled": False},
                "signal_export": {
                    "enabled": True,
                    "checkpoint": "dummy.pt",
                    "output_dir": "signals/test",
                },
                "backtesting": {"enabled": False},
            },
        }
        manifest_path = tmp_pipeline / "hft-ops" / "experiments" / "se_only.yaml"
        with open(manifest_path, "w") as f:
            yaml.dump(manifest_dict, f)

        manifest = load_manifest(manifest_path)
        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        result = validate_manifest(manifest, paths)

        no_stages_msgs = [
            str(w) for w in result.warnings if "No stages enabled" in str(w)
        ]
        assert not no_stages_msgs, (
            "signal_export-only manifest falsely flagged as empty: "
            f"{no_stages_msgs}"
        )

    def test_fully_disabled_still_warns(self, tmp_pipeline):
        """A manifest with ALL stages disabled should still surface the warning."""
        import yaml

        from hft_ops.manifest.loader import load_manifest
        from hft_ops.manifest.validator import validate_manifest
        from hft_ops.paths import PipelinePaths

        manifest_dict = {
            "experiment": {"name": "all_off", "contract_version": "2.2"},
            "pipeline_root": "..",
            "stages": {
                "extraction": {"enabled": False},
                "dataset_analysis": {"enabled": False},
                "validation": {"enabled": False},
                "training": {"enabled": False},
                "signal_export": {"enabled": False},
                "backtesting": {"enabled": False},
            },
        }
        manifest_path = tmp_pipeline / "hft-ops" / "experiments" / "off.yaml"
        with open(manifest_path, "w") as f:
            yaml.dump(manifest_dict, f)

        manifest = load_manifest(manifest_path)
        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        result = validate_manifest(manifest, paths)

        msgs = [str(w) for w in result.warnings if "No stages enabled" in str(w)]
        assert len(msgs) == 1


# -----------------------------------------------------------------------------
# C4: backfill duplicate benign-skip exits 0
# -----------------------------------------------------------------------------


class TestValidatorNoMutation:
    """validate_manifest must not mutate manifest state.

    The pre-fix code mutated ``stages.backtesting.horizon_idx``. Sweep
    expansion then validated per-grid-point in a loop, risking surprising
    side effects. The new invariant: validation is pure; the orchestrator
    applies resolved context explicitly via ``apply_resolved_context``.
    """

    def test_validate_manifest_does_not_mutate_horizon_idx(self, tmp_pipeline):
        """After validate_manifest, horizon_idx must remain at its input value."""
        import yaml

        from hft_ops.manifest.loader import load_manifest
        from hft_ops.manifest.validator import validate_manifest
        from hft_ops.paths import PipelinePaths

        extractor_toml = tmp_pipeline / "feature-extractor-MBO-LOB" / "configs" / "ext.toml"
        extractor_toml.write_text(
            """
[experiment]
name = "test"

[symbol]
name = "NVDA"
exchange = "XNAS"
filename_pattern = "xnas-itch-{date}.mbo.dbn.zst"
tick_size = 0.01

[data]
input_dir = "../data/NVDA"
output_dir = "../data/exports/test"

[features]
lob_levels = 10
include_derived = true
include_mbo = true
include_signals = true

[sampling]
strategy = "event_based"
event_count = 1000

[sequence]
window_size = 100
stride = 10

[labels]
strategy = "regression"
max_horizons = [10, 60, 300]
"""
        )

        trainer_yaml = tmp_pipeline / "lob-model-trainer" / "configs" / "experiments" / "t.yaml"
        trainer_yaml.parent.mkdir(parents=True, exist_ok=True)
        with open(trainer_yaml, "w") as f:
            yaml.dump({"data": {"feature_count": 98}}, f)

        manifest_dict = {
            "experiment": {"name": "no_mutation", "contract_version": "2.2"},
            "pipeline_root": "..",
            "stages": {
                "extraction": {
                    "enabled": True,
                    "config": "feature-extractor-MBO-LOB/configs/ext.toml",
                    "output_dir": "data/exports/test",
                },
                "training": {
                    "enabled": True,
                    "config": "lob-model-trainer/configs/experiments/t.yaml",
                    "horizon_value": 60,
                },
                "backtesting": {
                    "enabled": True,
                    "model_checkpoint": "dummy.pt",
                    "horizon_idx": None,  # would be mutated under old behavior
                },
            },
        }
        m_path = tmp_pipeline / "hft-ops" / "experiments" / "m.yaml"
        with open(m_path, "w") as f:
            yaml.dump(manifest_dict, f)

        manifest = load_manifest(m_path)
        assert manifest.stages.backtesting.horizon_idx is None

        paths = PipelinePaths(pipeline_root=tmp_pipeline)
        validate_manifest(manifest, paths)

        # PRE-FIX: validate_manifest would mutate horizon_idx to 1 (index of 60
        # in [10, 60, 300]). POST-FIX: it must remain None.
        assert manifest.stages.backtesting.horizon_idx is None, (
            "validate_manifest must not mutate backtesting.horizon_idx. "
            "Use resolve_manifest_context + apply_resolved_context instead."
        )

    def test_resolve_manifest_context_returns_horizon_idx(self, tmp_pipeline):
        """resolve_manifest_context translates horizon_value → horizon_idx."""
        import yaml

        from hft_ops.manifest.loader import load_manifest
        from hft_ops.manifest.validator import resolve_manifest_context
        from hft_ops.paths import PipelinePaths

        extractor_toml = tmp_pipeline / "feature-extractor-MBO-LOB" / "configs" / "ext.toml"
        extractor_toml.write_text(
            """
[experiment]
name = "test"

[symbol]
name = "NVDA"
exchange = "XNAS"
filename_pattern = "xnas-itch-{date}.mbo.dbn.zst"
tick_size = 0.01

[data]
input_dir = "../data/NVDA"
output_dir = "../data/exports/test"

[features]
lob_levels = 10
include_derived = true
include_mbo = true
include_signals = true

[sampling]
strategy = "event_based"
event_count = 1000

[sequence]
window_size = 100
stride = 10

[labels]
strategy = "regression"
max_horizons = [10, 60, 300]
"""
        )

        manifest_dict = {
            "experiment": {"name": "res_ctx", "contract_version": "2.2"},
            "pipeline_root": "..",
            "stages": {
                "extraction": {
                    "enabled": True,
                    "config": "feature-extractor-MBO-LOB/configs/ext.toml",
                    "output_dir": "data/exports/test",
                },
                "training": {
                    "enabled": True,
                    "trainer_config": {"model": {"model_type": "tlob"}},
                    "horizon_value": 60,
                },
                "backtesting": {"enabled": False},
            },
        }
        m_path = tmp_pipeline / "hft-ops" / "experiments" / "ctx.yaml"
        with open(m_path, "w") as f:
            yaml.dump(manifest_dict, f)

        manifest = load_manifest(m_path)
        paths = PipelinePaths(pipeline_root=tmp_pipeline)

        ctx = resolve_manifest_context(manifest, paths)
        assert ctx.horizon_value == 60
        assert ctx.horizon_idx == 1, f"expected idx 1 for value 60, got {ctx.horizon_idx}"
        assert ctx.feature_count == 98, (
            f"expected 98 features (10*4 + 8 + 36 + 14), got {ctx.feature_count}"
        )

    def test_apply_resolved_context_is_idempotent(self, tmp_pipeline):
        """apply_resolved_context must not overwrite an already-set horizon_idx."""
        import yaml

        from hft_ops.manifest.loader import load_manifest
        from hft_ops.manifest.validator import (
            ResolvedContext,
            apply_resolved_context,
        )

        manifest_dict = {
            "experiment": {"name": "idempotent", "contract_version": "2.2"},
            "pipeline_root": "..",
            "stages": {
                "extraction": {"enabled": False},
                "training": {"enabled": False},
                "backtesting": {
                    "enabled": True,
                    "model_checkpoint": "dummy.pt",
                    "horizon_idx": 5,  # user explicitly set
                },
            },
        }
        m_path = tmp_pipeline / "hft-ops" / "experiments" / "idem.yaml"
        with open(m_path, "w") as f:
            yaml.dump(manifest_dict, f)

        manifest = load_manifest(m_path)
        assert manifest.stages.backtesting.horizon_idx == 5

        ctx = ResolvedContext(horizon_idx=99)  # would overwrite if non-idempotent
        apply_resolved_context(manifest, ctx)

        assert manifest.stages.backtesting.horizon_idx == 5, (
            "apply_resolved_context must NOT overwrite explicit user-set value"
        )


class TestBackfillDuplicateBenignSkip:
    """C4 regression test — snapshot test for the source-code invariant.

    A full end-to-end CLI test requires a full pipeline fixture; this
    lightweight snapshot test guards against reintroduction of the exit-1
    behavior by reading the source file directly.
    """

    def test_duplicate_exit_code_is_zero(self):
        """The duplicate-skip code path must use sys.exit(0), not 1."""
        cli_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "hft_ops"
            / "cli.py"
        )
        source = cli_path.read_text()

        # Locate the backfill duplicate-skip block; must exit 0 not 1.
        assert "Skipping; record not re-registered" in source, (
            "Expected backfill skip message missing — has the backfill CLI "
            "been renamed or refactored? Update this test."
        )
        # Ensure the block context says exit(0).
        # We do NOT want to match the sweep or run commands' duplicate handling.
        skip_idx = source.index("Skipping; record not re-registered")
        # Look ahead 400 chars for the sys.exit call.
        window = source[skip_idx : skip_idx + 400]
        assert "sys.exit(0)" in window, (
            "Backfill duplicate skip must exit(0) (benign skip), not exit(1). "
            "Regression of C4."
        )
        assert "sys.exit(1)" not in window, (
            "Backfill duplicate skip must NOT exit(1). Regression of C4."
        )
