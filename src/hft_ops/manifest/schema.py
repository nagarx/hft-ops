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

    Args:
        enabled: Whether to run this stage.
        config: Path to the trainer YAML config, relative to pipeline_root.
        overrides: Key-value pairs to override in the trainer config. Supports
            dotted keys (e.g., "data.data_dir").
        horizon_value: Explicit horizon value (e.g., 100). Resolved to
            horizon_idx at runtime from the export's horizons metadata.
        output_dir: Output directory for training artifacts.
        extra_args: Additional CLI arguments to pass to train.py.
    """

    enabled: bool = True
    config: str = ""
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
        model_checkpoint: Path to the model checkpoint file.
        data_dir: Path to feature exports for backtesting.
        horizon_idx: Horizon index for backtesting. Use "${resolved.horizon_idx}"
            to auto-resolve from the training stage.
        params: Backtest parameters.
        extra_args: Additional CLI arguments to pass to backtest script.
    """

    enabled: bool = True
    model_checkpoint: str = ""
    data_dir: str = ""
    horizon_idx: Optional[int] = None
    params: BacktestParams = field(default_factory=BacktestParams)
    extra_args: List[str] = field(default_factory=list)


@dataclass
class Stages:
    """Container for all pipeline stages."""

    extraction: ExtractionStage = field(default_factory=ExtractionStage)
    raw_analysis: RawAnalysisStage = field(default_factory=RawAnalysisStage)
    dataset_analysis: DatasetAnalysisStage = field(
        default_factory=DatasetAnalysisStage
    )
    training: TrainingStage = field(default_factory=TrainingStage)
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
