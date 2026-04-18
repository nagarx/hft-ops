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
        skip_if_exists: Reuse existing exports if output_dir exists
            and metadata validates against the contract.
        config: Path to the extractor TOML config, relative to pipeline_root.
        output_dir: Path for exported data, relative to pipeline_root.
    """

    enabled: bool = True
    skip_if_exists: bool = True
    config: str = ""
    output_dir: str = ""


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

    All values have sensible defaults that can be overridden per-experiment.
    """

    initial_capital: float = 100_000.0
    position_size: float = 0.1
    spread_bps: float = 1.0
    slippage_bps: float = 0.5
    threshold: float = 0.0
    no_short: bool = False
    device: str = "cpu"


@dataclass
class BacktestingStage:
    """Backtest stage configuration.

    Args:
        enabled: Whether to run this stage.
        script: Backtest script to invoke, relative to backtester_dir.
            Defaults to ``scripts/backtest_deeplob.py``. Other supported scripts:
            ``scripts/run_readability_backtest.py``, ``scripts/run_regression_backtest.py``,
            ``scripts/run_spread_signal_backtest.py``. Script-specific args are
            passed via ``extra_args`` or ``params_file``.
        model_checkpoint: Path to the model checkpoint file.
        data_dir: Path to feature exports for backtesting.
        signals_dir: Path to signals output from SignalExportStage (for
            signal-based backtests like readability / regression).
        horizon_idx: Horizon index for backtesting. Use "${resolved.horizon_idx}"
            to auto-resolve from the training stage.
        params: Backtest parameters (for backtest_deeplob.py compatibility).
        params_file: Optional path to a script-specific YAML config (passed
            via ``--params-file`` to scripts that support it).
        extra_args: Additional CLI arguments to pass to backtest script.
    """

    enabled: bool = True
    script: str = "scripts/backtest_deeplob.py"
    model_checkpoint: str = ""
    data_dir: str = ""
    signals_dir: str = ""
    horizon_idx: Optional[int] = None
    params: BacktestParams = field(default_factory=BacktestParams)
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

    Invokes a trainer-side signal export script (e.g., ``export_signals.py``,
    ``export_hmhp_signals.py``) to materialize predictions from a checkpoint
    into a ``signals/`` directory that the backtester consumes.

    Runs BETWEEN training and backtesting. If disabled, backtesting stage
    must either reuse pre-existing signals or run its own export.

    Args:
        enabled: Whether to run this stage.
        script: Signal export script, relative to trainer_dir.
        checkpoint: Path to the trained model checkpoint.
        split: Dataset split to export signals for (train | val | test).
        output_dir: Where to write signals/*.npy files.
        extra_args: Additional CLI arguments to pass to the script.
    """

    enabled: bool = False
    script: str = "scripts/export_signals.py"
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
