"""
Experiment manifest dataclasses.

An ExperimentManifest is the single YAML file that defines a complete
experiment: which data to extract, which model to train, which backtest
to run, and how to track results. It references existing module configs
but NEVER duplicates parameters across stages.

Schema version: 1.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentHeader:
    """Top-level experiment metadata."""

    name: str
    description: str = ""
    hypothesis: str = ""
    contract_version: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class ExtractionStage:
    """Feature extraction stage configuration.

    Args:
        enabled: Whether to run this stage.
        skip_if_exists: **DEPRECATED as of Phase 8A.0 (2026-04-20)** — legacy
            filesystem-existence check superseded by the content-addressed
            extraction cache at ``data/exports/_cache/``. The cache covers
            config-changes / git-SHA-changes / raw-input-changes that the
            filename-based ``skip_if_exists`` missed, eliminating a class
            of silent-correctness bugs where the extractor was SKIPPED but
            the config had changed in a breaking way. Still honoured as a
            fallback when cache-key inputs cannot be gathered (e.g., in
            test environments without git repos). Removal deadline:
            Phase 9 (when SQLite ledger migration retires the JSON
            envelope; target Q3 2026). Emits ``DeprecationWarning`` on
            manifest load when explicitly set to a non-default value.
        config: Path to the extractor TOML config, relative to pipeline_root.
        output_dir: Path for exported data, relative to pipeline_root.
    """

    enabled: bool = True
    skip_if_exists: bool = True
    config: str = ""
    output_dir: str = ""

    def __post_init__(self) -> None:
        # Phase 8A.0 (2026-04-20): warn only when user *explicitly* set
        # ``skip_if_exists: false`` (the default True matches legacy
        # behavior and the cache supersedes it silently). We cannot tell
        # "default True" from "explicit True" at dataclass construction
        # time, so only the False-case triggers the warning (signals
        # intentional legacy opt-in that is now redundant).
        if self.skip_if_exists is False:
            import warnings

            warnings.warn(
                "ExtractionStage.skip_if_exists=false is deprecated (Phase "
                "8A.0). The content-addressed extraction cache at "
                "data/exports/_cache/ supersedes it with stricter "
                "invalidation (config + git + raw-input + binary + "
                "platform changes all invalidate). To disable caching "
                "entirely pass --no-cache-extraction to `hft-ops run`. "
                "Removal target: Phase 9 (Q3 2026).",
                DeprecationWarning,
                stacklevel=3,
            )


@dataclass
class RawAnalysisStage:
    """MBO-LOB-analyzer stage configuration.

    Args:
        enabled: Whether to run this stage.
        profile: Analysis profile name (quick, standard, full, etc.).
        symbol: Ticker symbol for the analysis.
        data_dir: Path to raw Parquet exports. If empty, inferred from
            the reconstructor output referenced in the extraction config.
        analyzers: Explicit analyzer list (overrides profile).
    """

    enabled: bool = False
    profile: str = "standard"
    symbol: str = ""
    data_dir: str = ""
    analyzers: List[str] = field(default_factory=list)
    output_dir: str = ""


@dataclass
class DatasetAnalysisStage:
    """lob-dataset-analyzer stage configuration.

    Args:
        enabled: Whether to run this stage.
        profile: Analysis profile name (quick, full, volatility, etc.).
        split: Dataset split to analyze (train, val, test).
        data_dir: Overrides extraction output_dir if set.
        analyzers: Explicit analyzer list (overrides profile).
    """

    enabled: bool = True
    profile: str = "quick"
    split: str = "train"
    data_dir: str = ""
    analyzers: List[str] = field(default_factory=list)
    output_dir: str = ""


@dataclass
class TrainingStage:
    """Model training stage configuration.

    The trainer reads feature_count, window_size, stride from export metadata
    at load time -- NOT from its own config. The trainer YAML specifies only
    training-specific params (model, optimizer, epochs, etc.).

    The trainer config can be specified via EITHER:
      - ``config``: path to an existing trainer YAML file (legacy path)
      - ``trainer_config``: inline dict embedded in the manifest (unified path,
        introduced in Phase 1 of the training-pipeline-architecture migration)

    When both are set, the loader raises; when neither is set (and enabled=True),
    the validator raises. The unified/inline form is the preferred pattern for
    new manifests — it keeps the experiment definition in a single file and
    eliminates the 2-files-per-experiment duplication. Inline ``_base:`` is
    supported (resolved relative to pipeline_root for Phase 3 OmegaConf composition).

    Args:
        enabled: Whether to run this stage.
        config: Path to the trainer YAML config, relative to pipeline_root.
            Mutually exclusive with ``trainer_config``.
        trainer_config: Inline trainer configuration dict. Mutually exclusive
            with ``config``. The runner materializes this to a temp YAML file
            at runtime, which train.py then consumes.
        overrides: Key-value pairs to override in the trainer config. Supports
            dotted keys (e.g., "data.data_dir"). Applied AFTER inheritance
            resolution, regardless of whether ``config`` or ``trainer_config``
            is the source.
        horizon_value: Explicit horizon value (e.g., 100). Resolved to
            horizon_idx at runtime from the export's horizons metadata.
        output_dir: Output directory for training artifacts.
        extra_args: Additional CLI arguments to pass to train.py.
    """

    enabled: bool = True
    config: str = ""
    trainer_config: Optional[Dict[str, Any]] = None
    overrides: Dict[str, Any] = field(default_factory=dict)
    horizon_value: Optional[int] = None
    output_dir: str = ""
    extra_args: List[str] = field(default_factory=list)


@dataclass
class BacktestParams:
    """Backtest parameter set.

    Only ``initial_capital``, ``position_size``, and ``spread_bps`` are passed
    to the current backtester scripts (readability / regression / spread_signal).
    All other fields are DEPRECATED — see per-field notes. Removal deadline
    for deprecated fields: 2026-10-31.

    All values have sensible defaults that can be overridden per-experiment.

    Args:
        initial_capital: Starting capital for backtest (USD). Passed as
            ``--initial-capital``.
        position_size: Position size as fraction of capital (0.0-1.0).
            Passed as ``--position-size``.
        spread_bps: Maximum bid-ask spread in basis points for entry. Passed
            as ``--max-spread-bps`` (Phase 7.5-B.1 renamed from `--spread-bps`
            to match current backtester script argparse).
        slippage_bps: DEPRECATED (Phase 7.5-B.1, 2026-04-23) — no current
            backtester script accepts ``--slippage-bps``. Field retained for
            back-compat with 2 existing manifests (`nvda_hmhp_tb_volscaled.yaml`,
            `e5_60s_importance_audit.yaml`). Runner no longer passes this flag.
            Removal: 2026-10-31. Migration: slippage is now built into the
            IBKR-calibrated cost model inside scripts.
        threshold: DEPRECATED — master-backtester-era field. No current
            script accepts ``--threshold``. Removal: 2026-10-31.
        no_short: DEPRECATED — master-backtester-era field. No current
            script accepts ``--no-short``. Removal: 2026-10-31.
        device: DEPRECATED — master-backtester-era field. Backtester scripts
            run CPU-only (numpy-vectorized). No ``--device`` flag.
            Removal: 2026-10-31.
    """

    initial_capital: float = 100_000.0
    position_size: float = 0.1
    spread_bps: float = 1.0
    # DEPRECATED as of Phase 7.5-B.1 (2026-04-23); removal 2026-10-31
    slippage_bps: float = 0.5
    # DEPRECATED — no backtester script accepts --threshold; removal 2026-10-31
    threshold: float = 0.0
    # DEPRECATED — no backtester script accepts --no-short; removal 2026-10-31
    no_short: bool = False
    # DEPRECATED — backtester scripts run CPU-only; removal 2026-10-31
    device: str = "cpu"


@dataclass
class BacktestingStage:
    """Backtest stage configuration.

    Phase V.1.5 Frame-5 Task-1c Bug #5 closure (Phase 7.5-B.1, 2026-04-23):
    BacktestRunner previously constructed a cmd with many flags that NO
    current backtester script accepts. The runner was designed against a
    pre-existing "master backtester" that was refactored into 3 specialized
    scripts (readability / regression / spread_signal) without updating the
    runner's arg protocol. Several fields on this dataclass are now
    OPERATIONALLY DEAD (runner no longer passes them to subprocess) and
    marked DEPRECATED for removal 2026-10-31. See per-field notes.

    For script-specific flags (`--min-agreement` readability, `--zero-dte`
    regression, `--exchange ARCX`, `--commission`, `--hold-events`, etc.),
    use the ``extra_args`` field — that is the documented escape hatch for
    passing any flag the shared cmd construction doesn't surface.

    Args:
        enabled: Whether to run this stage.
        script: Backtest script path, RELATIVE TO PIPELINE ROOT (not
            backtester_dir — Phase V.1.5 Frame-5 Task-1c unified the
            script-path convention). Defaults to
            ``scripts/backtest_deeplob.py`` for back-compat; production
            manifests should use one of:
            ``lob-backtester/scripts/run_readability_backtest.py`` (HMHP
            classification + confirmation gate),
            ``lob-backtester/scripts/run_regression_backtest.py`` (TLOB /
            HMHP-R regression), or
            ``lob-backtester/scripts/run_spread_signal_backtest.py``
            (spread-based signals, NON-ORCHESTRATABLE — see docstring).
        signals_dir: Path to signals output from SignalExportStage (for
            signal-based backtests). Runner passes this as ``--signals``
            (renamed from ``--signals-dir`` in Phase 7.5-B.1).
        params: Backtest parameters (initial_capital, position_size,
            spread_bps). See `BacktestParams` for per-field notes.
        extra_args: Additional CLI arguments to pass to backtest script
            (e.g., `["--min-agreement", "1.0"]` for readability-specific
            flags). THIS IS THE DOCUMENTED ESCAPE HATCH for flags not
            surfaced by the shared cmd construction.

        model_checkpoint: DEPRECATED (Phase 7.5-B.1, 2026-04-23) — no
            current backtester script accepts ``--model-checkpoint``.
            Signal-based backtests read from `signals_dir` (trainer
            already ran inference into the signal files during
            SignalExportStage). Field retained for back-compat; removal
            2026-10-31.
        data_dir: DEPRECATED — no current script accepts ``--data-dir``.
            Legacy master-backtester artifact. Removal 2026-10-31.
        horizon_idx: DEPRECATED — no current script accepts
            ``--horizon-idx``. Use ``extra_args: ["--primary-horizon-idx", "0"]``
            if a script supports horizon selection (readability accepts
            ``--primary-horizon-idx``). Removal 2026-10-31.
        params_file: DEPRECATED — no current script accepts
            ``--params-file``. YAML params should be embedded in manifest
            or passed explicitly via ``extra_args``. Removal 2026-10-31.
    """

    enabled: bool = True
    script: str = "scripts/backtest_deeplob.py"
    # DEPRECATED as of Phase 7.5-B.1 (2026-04-23); removal 2026-10-31
    model_checkpoint: str = ""
    # DEPRECATED — no backtester script accepts --data-dir; removal 2026-10-31
    data_dir: str = ""
    signals_dir: str = ""
    # DEPRECATED — use extra_args=["--primary-horizon-idx",...] instead
    # Removal 2026-10-31
    horizon_idx: Optional[int] = None
    params: BacktestParams = field(default_factory=BacktestParams)
    # DEPRECATED — no backtester script accepts --params-file; removal 2026-10-31
    params_file: str = ""
    extra_args: List[str] = field(default_factory=list)


@dataclass
class ValidationStage:
    """Pre-training IC-gate validation stage (Phase 2).

    Runs between ``dataset_analysis`` and ``training``. Computes per-feature
    Spearman IC against the target label, checks Rule-13 decision gates
    (IC > 0.05, IC_COUNT >= 2, return_std > 5 bps, walk-forward stability > 2.0),
    and emits a ``gate_report.json`` artifact. On failure, behavior is
    controlled by ``on_fail``:

    - ``warn`` (DEFAULT): log the failure, record it in the ledger, but
      proceed to training. Rationale: the evaluator CLAUDE.md explicitly
      warns against using DISCARD as a hard gate — individual-feature IC
      misses interaction / temporal / context-feature value. We want the
      gate to SURFACE failures so researchers can investigate, not silently
      block valid experiments.
    - ``abort``: raise StageFailure → pipeline stops → ledger record saved
      with status=failed. Use after confidence is established.
    - ``record_only``: record the gate outcome but never fail. For archival /
      post-hoc exploration where the gate's verdict is informational only.

    Attributes:
        enabled: Whether to run the gate.
        on_fail: Action on gate failure. See above.
        target_horizon: Horizon label (e.g., ``"H10"``, ``"10"``). Empty →
            auto-infer from ``training.horizon_value``.
        min_ic: G_IC threshold (best absolute feature IC).
        min_ic_count: G_IC_COUNT threshold (#features with |IC| > min_ic).
        min_return_std_bps: G_RETURN_STD threshold (label std in bps).
        min_stability: G_STABILITY threshold (walk-forward mean/std ratio).
        sample_size: Max sequences sampled from train for IC estimation.
        n_folds: Walk-forward fold count; adaptively clipped at runtime
            (``max(5, min(n_folds, train_days // 8))``).
        allow_zero_ic_names: Feature names that bypass the IC check
            (context features — dark_share, time_regime, etc.).
        profile_ref: Optional path to a precomputed ``feature_profiles.json``
            from a full evaluator run; gate post-processes that rather than
            recomputing.
        output_dir: Where to write ``gate_report.json`` / ``per_feature_ic.csv``.
            Empty → defaults to the experiment's runs directory.
    """

    enabled: bool = True
    on_fail: str = "warn"
    target_horizon: str = ""
    min_ic: float = 0.05
    min_ic_count: int = 2
    min_return_std_bps: float = 5.0
    min_stability: float = 2.0
    sample_size: int = 200_000
    n_folds: int = 20
    allow_zero_ic_names: List[str] = field(default_factory=list)
    profile_ref: str = ""
    output_dir: str = ""


@dataclass
class PostTrainingGateStage:
    """Post-training regression-detection gate (Phase 7 Stage 7.4).

    Runs AFTER ``training`` and BEFORE ``signal_export``. Compares the
    just-completed experiment's metrics against:

    1. **Floor check**: is the primary metric (IC for regression, macro_f1
       for classification) above an absolute floor? Catches degenerate
       runs (e.g., mean-collapse, loss inversion).
    2. **Prior-best ratio check**: is the primary metric within
       ``min_ratio_vs_prior_best`` of the best prior experiment for the
       SAME (model_type, label_type, horizon) signature? Catches silent
       regressions as the pipeline evolves.
    3. **Cost-breakeven check** (optional, regression only): is the
       model's prediction magnitude plausibly tradable against IBKR-
       calibrated cost floors? Reports only (no abort default).

    On regression (any check fails), behavior is controlled by
    ``on_regression``:

    - ``warn`` (DEFAULT): log + record gate_report.json in the output
      dir + attach summary to ExperimentRecord.notes + PROCEED to
      signal_export. Rationale: a researcher often still wants to see
      the full backtest + signal outputs when a regression is flagged,
      to understand WHY. Aborting loses diagnostic information.
    - ``abort``: raise StageFailure → pipeline stops → ledger record
      saved with ``status: failed`` and the full gate_report attached.
      Use in CI / auto-sweep contexts where regressions should halt
      computation.
    - ``record_only``: record the gate outcome but never fail, never
      warn. For archival / post-hoc analysis where the gate is
      informational only.

    Attributes:
        enabled: Whether to run the gate. Default False (opt-in during
            Phase 7.4 rollout; may default True in Phase 8 after
            validation).
        on_regression: Disposition on regression detection.
        primary_metric: Which captured_metric to use for comparisons.
            Common values: ``"test_ic"`` (regression), ``"best_val_macro_f1"``
            (classification), ``"test_directional_accuracy"``. Empty →
            auto-infer (test_ic > test_directional_accuracy > best_val_macro_f1).
        min_metric_floor: Floor value for the primary metric. Default
            0.05 matches the pre-training IC floor (Rule 13).
        min_ratio_vs_prior_best: New metric must be >= this ratio of the
            best prior experiment's same metric. 0.9 means "within 10%
            of prior best". Set to 0 to disable the ratio check.
        match_on_signature: Tuple of fields used to match prior experiments
            for the ratio check. Default ``("model_type", "labeling_strategy",
            "horizon_value")``. Experiments whose signature differs are
            excluded from the ratio check.
        cost_breakeven_bps: Cost floor in basis points (IBKR-calibrated
            Deep ITM = 1.4 bps). 0 disables the check.
        output_dir: Where to write ``gate_report.json``. Empty → defaults
            to the experiment's runs directory.
    """

    enabled: bool = False
    on_regression: str = "warn"
    primary_metric: str = ""
    min_metric_floor: float = 0.05
    min_ratio_vs_prior_best: float = 0.9
    match_on_signature: List[str] = field(
        default_factory=lambda: ["model_type", "labeling_strategy", "horizon_value"]
    )
    cost_breakeven_bps: float = 1.4
    output_dir: str = ""


@dataclass
class SignalExportStage:
    """Signal export stage configuration.

    Invokes ``lob-model-trainer/scripts/export_signals.py`` to materialize
    predictions from a trained checkpoint into a ``signals/`` directory that
    the backtester consumes. Phase 6 6D (2026-04-17) archived the legacy per-
    model scripts (``export_hmhp_signals.py``, ``export_regression_signals.py``,
    ``export_tlob_regression_signals.py``) in favor of the unified
    ``export_signals.py`` entry point.

    Runs BETWEEN training and backtesting. If disabled, backtesting stage
    must either reuse pre-existing signals or run its own export.

    Config resolution (Phase 7.5-A, 2026-04-23): ``export_signals.py``
    requires ``--config <trainer_yaml_path>`` to reconstruct the Trainer
    pipeline (normalization, feature selection, model instantiation). The
    SignalExportRunner resolves this path via a 3-tier fallback:

      1. **Priority 1** (new-style wrapper-less manifest): if the trainer
         stage ran with an inline ``trainer_config:`` dict, ``train.py``
         persists the effective config to ``<training.output_dir>/config.yaml``
         (Phase 7.5-A+ train.py extension). SignalExportRunner uses this path
         by default.
      2. **Priority 2** (legacy wrapper manifest): if the manifest declares
         ``stages.training.config: <path>`` AND Priority 1 file does not yet
         exist, fall back to ``manifest.stages.training.config`` (resolved
         via ``config.paths.resolve``).
      3. **Priority 3** (explicit escape hatch): ``SignalExportStage.config``
         when set takes precedence over BOTH fallbacks. Use this when the
         signal-export config must differ from training (e.g., different
         split, different calibration settings).

    Validator fails-loud if signal_export.enabled AND none of the 3
    sources resolve to an existing file.

    Args:
        enabled: Whether to run this stage.
        script: Signal export script path, RELATIVE TO PIPELINE ROOT
            (e.g., ``"lob-model-trainer/scripts/export_signals.py"``). Phase
            V.1.5 Frame-5 Task-1c (2026-04-23) unified script-path resolution
            across signal_export + backtesting stages to pipeline-root-
            relative per the convention used by ``extraction.config`` +
            ``data.data_dir`` + ``stage.checkpoint``.
        config: Trainer YAML config path (Phase 7.5-A escape hatch). When
            set, takes precedence over ``manifest.stages.training.config``
            and the auto-persisted ``<training.output_dir>/config.yaml``.
            Typically UNSET — operators rely on automatic resolution.
        checkpoint: Path to the trained model checkpoint.
        split: Dataset split to export signals for (val | test). ``train``
            is NOT accepted by ``export_signals.py`` (Phase 7.5-A fix).
        output_dir: Where to write signals/*.npy files.
        extra_args: Additional CLI arguments to pass to the script (e.g.,
            ``["--calibrate", "variance_match"]`` for regression
            calibration, ``["--batch-size", "64"]`` for memory override).
    """

    enabled: bool = False
    script: str = "scripts/export_signals.py"
    config: Optional[str] = None
    checkpoint: str = ""
    split: str = "test"
    output_dir: str = ""
    extra_args: List[str] = field(default_factory=list)


@dataclass
class Stages:
    """Container for all pipeline stages.

    Stage order (as executed by the runner in cli.py):
        extraction → raw_analysis → dataset_analysis → validation
            → training → signal_export → backtesting
    """

    extraction: ExtractionStage = field(default_factory=ExtractionStage)
    raw_analysis: RawAnalysisStage = field(default_factory=RawAnalysisStage)
    dataset_analysis: DatasetAnalysisStage = field(
        default_factory=DatasetAnalysisStage
    )
    validation: ValidationStage = field(default_factory=ValidationStage)
    training: TrainingStage = field(default_factory=TrainingStage)
    # Phase 7 Stage 7.4 (2026-04-19): post-training regression-detection gate.
    # Runs between training and signal_export. Default enabled=False — opt-in
    # during Phase 7.4 rollout. See PostTrainingGateStage docstring.
    post_training_gate: PostTrainingGateStage = field(
        default_factory=PostTrainingGateStage
    )
    signal_export: SignalExportStage = field(default_factory=SignalExportStage)
    backtesting: BacktestingStage = field(default_factory=BacktestingStage)


# =============================================================================
# Sweep / Grid Expansion (Phase 4)
# =============================================================================


@dataclass
class SweepAxisValue:
    """One value along a sweep axis.

    Attributes:
        label: Short label for this value (used in experiment naming).
            Must match [a-zA-Z0-9_-]+.
        overrides: Dotted-key overrides applied to the manifest/trainer config.
            Manifest-level keys (horizon_value, output_dir, etc.) are applied
            to the TrainingStage dataclass. All other dotted keys are merged
            into stages.training.overrides for trainer YAML modification.
    """

    label: str = ""
    overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SweepAxis:
    """One dimension of a parameter sweep.

    Attributes:
        name: Axis name for identification and reporting (e.g., "model", "horizon").
        values: List of values to sweep over. Each produces one grid point per
            combination with other axes.
    """

    name: str = ""
    values: List[SweepAxisValue] = field(default_factory=list)


@dataclass
class SweepConfig:
    """Configuration for a parameter sweep / grid search.

    Defines axes to vary and the expansion strategy. The Cartesian product
    of all axes produces N concrete experiments, each with its own set of
    overrides applied on top of the base manifest.

    Attributes:
        name: Sweep name (used as prefix for generated experiment names).
        strategy: Expansion strategy. Currently only "grid" (Cartesian product).
        axes: List of sweep axes. Order determines label concatenation order.
    """

    name: str = ""
    strategy: str = "grid"
    axes: List[SweepAxis] = field(default_factory=list)


@dataclass
class ExperimentManifest:
    """Complete experiment definition.

    A single YAML file defines the entire experiment pipeline. The manifest
    references existing module configs and resolves cross-stage dependencies
    through ${...} variable substitution.

    When the ``sweep`` field is populated, this manifest is a sweep template
    that produces N concrete experiments via ``expand_sweep()``. Use
    ``hft-ops sweep run`` instead of ``hft-ops run`` for sweep manifests.

    Attributes:
        experiment: Experiment metadata (name, description, hypothesis, tags).
        pipeline_root: Path to HFT-pipeline-v2 root, relative to manifest file.
        stages: Configuration for each pipeline stage.
        sweep: Optional sweep configuration for grid/parameter searches.
        manifest_path: Absolute path to the source YAML file (set by loader).
    """

    experiment: ExperimentHeader = field(default_factory=ExperimentHeader)
    pipeline_root: str = ".."
    stages: Stages = field(default_factory=Stages)
    sweep: Optional[SweepConfig] = None
    manifest_path: str = ""
