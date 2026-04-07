"""
hft-ops CLI: Central experiment orchestrator for the HFT pipeline.

Usage:
    hft-ops run <manifest.yaml>          Run a complete experiment
    hft-ops validate <manifest.yaml>     Validate without executing
    hft-ops compare                      Compare experiments by metrics
    hft-ops diff <id1> <id2>             Detailed diff between two experiments
    hft-ops ledger list                  List all experiments
    hft-ops ledger show <id>             Show full experiment record
    hft-ops ledger search                Search experiments by criteria
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from hft_ops.config import OpsConfig
from hft_ops.ledger.comparator import compare_experiments, diff_experiments
from hft_ops.ledger.dedup import check_duplicate, compute_fingerprint
from hft_ops.ledger.experiment_record import ExperimentRecord
from hft_ops.ledger.ledger import ExperimentLedger
from hft_ops.manifest.loader import load_manifest
from hft_ops.manifest.validator import validate_manifest
from hft_ops.paths import PipelinePaths
from hft_ops.provenance.lineage import build_provenance
from hft_ops.stages.backtesting import BacktestRunner
from hft_ops.stages.base import StageResult, StageStatus
from hft_ops.stages.dataset_analysis import DatasetAnalysisRunner
from hft_ops.stages.extraction import ExtractionRunner
from hft_ops.stages.raw_analysis import RawAnalysisRunner
from hft_ops.stages.training import TrainingRunner

console = Console()


def _resolve_pipeline_root(ctx_root: Optional[str]) -> Path:
    """Resolve pipeline root from CLI option or auto-detect."""
    if ctx_root:
        return Path(ctx_root).resolve()
    try:
        return PipelinePaths.auto_detect().pipeline_root
    except FileNotFoundError:
        console.print(
            "[red]Cannot auto-detect pipeline root. "
            "Use --pipeline-root to specify it.[/red]"
        )
        sys.exit(1)


@click.group()
@click.option(
    "--pipeline-root",
    type=click.Path(exists=True),
    default=None,
    help="Path to HFT-pipeline-v2 root. Auto-detected if not specified.",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose subprocess output.")
@click.pass_context
def main(ctx: click.Context, pipeline_root: Optional[str], verbose: bool) -> None:
    """hft-ops: Central experiment orchestrator for the HFT pipeline."""
    ctx.ensure_object(dict)
    ctx.obj["pipeline_root"] = pipeline_root
    ctx.obj["verbose"] = verbose


@main.command()
@click.argument("manifest_path", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Validate without executing stages.")
@click.option(
    "--stages",
    type=str,
    default=None,
    help="Comma-separated stages to run (e.g., extraction,training).",
)
@click.option("--resume", is_flag=True, help="Skip already-completed stages.")
@click.option("--force", is_flag=True, help="Run even if duplicate fingerprint found.")
@click.pass_context
def run(
    ctx: click.Context,
    manifest_path: str,
    dry_run: bool,
    stages: Optional[str],
    resume: bool,
    force: bool,
) -> None:
    """Run a complete experiment from a manifest YAML."""
    pipeline_root = _resolve_pipeline_root(ctx.obj.get("pipeline_root"))
    paths = PipelinePaths(pipeline_root=pipeline_root)
    ops_config = OpsConfig(
        paths=paths,
        verbose=ctx.obj.get("verbose", False),
        dry_run=dry_run,
    )

    console.print(f"[bold]Loading manifest:[/bold] {manifest_path}")
    manifest = load_manifest(manifest_path)

    if manifest.sweep is not None:
        console.print(
            "[red bold]This manifest contains a sweep section. "
            "Use 'hft-ops sweep run' instead of 'hft-ops run'.[/red bold]"
        )
        sys.exit(1)

    console.print(f"[bold]Validating...[/bold]")
    validation = validate_manifest(manifest, paths)

    for issue in validation.issues:
        color = "red" if issue.severity == "error" else "yellow"
        console.print(f"  [{color}]{issue}[/{color}]")

    if not validation.is_valid:
        console.print("[red bold]Validation failed. Aborting.[/red bold]")
        sys.exit(1)

    if validation.warnings:
        console.print(
            f"[yellow]{len(validation.warnings)} warnings (continuing)[/yellow]"
        )

    fingerprint = compute_fingerprint(manifest, paths)
    console.print(f"[dim]Fingerprint: {fingerprint[:16]}...[/dim]")

    if not dry_run:
        existing = check_duplicate(fingerprint, paths.ledger_dir)
        if existing and not force:
            console.print(
                f"[yellow bold]Duplicate experiment found:[/yellow bold] "
                f"{existing.get('experiment_id')}"
            )
            console.print(
                "Use --force to run anyway, or check the ledger."
            )
            sys.exit(0)

    requested_stages = None
    if stages:
        requested_stages = set(stages.split(","))

    stage_runners = [
        ("extraction", manifest.stages.extraction.enabled, ExtractionRunner()),
        ("raw_analysis", manifest.stages.raw_analysis.enabled, RawAnalysisRunner()),
        ("dataset_analysis", manifest.stages.dataset_analysis.enabled, DatasetAnalysisRunner()),
        ("training", manifest.stages.training.enabled, TrainingRunner()),
        ("backtesting", manifest.stages.backtesting.enabled, BacktestRunner()),
    ]

    results: dict[str, StageResult] = {}
    total_start = time.monotonic()

    for stage_name, enabled, runner in stage_runners:
        if not enabled:
            continue
        if requested_stages and stage_name not in requested_stages:
            continue

        console.print(f"\n[bold cyan]--- Stage: {stage_name} ---[/bold cyan]")

        input_errors = runner.validate_inputs(manifest, ops_config)
        if input_errors:
            for err in input_errors:
                console.print(f"  [red]{err}[/red]")
            results[stage_name] = StageResult(
                stage_name=stage_name,
                status=StageStatus.FAILED,
                error_message="; ".join(input_errors),
            )
            console.print(f"[red]Stage {stage_name} failed input validation.[/red]")
            break

        result = runner.run(manifest, ops_config)
        results[stage_name] = result

        status_color = {
            StageStatus.COMPLETED: "green",
            StageStatus.SKIPPED: "yellow",
            StageStatus.FAILED: "red",
        }.get(result.status, "white")

        console.print(
            f"  Status: [{status_color}]{result.status.value}[/{status_color}] "
            f"({result.duration_seconds:.1f}s)"
        )

        if result.error_message:
            console.print(f"  [red]{result.error_message}[/red]")

        if result.status == StageStatus.FAILED:
            console.print(f"[red bold]Pipeline aborted at stage: {stage_name}[/red bold]")
            break

    total_duration = time.monotonic() - total_start

    if dry_run:
        console.print("\n[yellow bold]Dry run complete. No stages executed.[/yellow bold]")
        return

    console.print(f"\n[bold]--- Recording experiment ---[/bold]")
    _record_experiment(manifest, paths, fingerprint, results, total_duration)


def _record_experiment(
    manifest,
    paths: PipelinePaths,
    fingerprint: str,
    results: dict[str, StageResult],
    total_duration: float,
) -> None:
    """Build and store an ExperimentRecord from stage results."""
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%S")
    experiment_id = f"{manifest.experiment.name}_{timestamp}_{fingerprint[:8]}"

    ext_config_path = None
    if manifest.stages.extraction.config:
        ext_config_path = paths.resolve(manifest.stages.extraction.config)

    train_config_path = None
    if manifest.stages.training.config:
        train_config_path = paths.resolve(manifest.stages.training.config)

    provenance = build_provenance(
        paths.pipeline_root,
        manifest_path=Path(manifest.manifest_path) if manifest.manifest_path else None,
        extractor_config_path=ext_config_path,
        trainer_config_path=train_config_path,
        data_dir=(
            paths.resolve(manifest.stages.extraction.output_dir)
            if manifest.stages.extraction.output_dir
            else None
        ),
        contract_version=manifest.experiment.contract_version,
    )

    stages_completed = [
        name for name, r in results.items()
        if r.status in (StageStatus.COMPLETED, StageStatus.SKIPPED)
    ]

    any_failed = any(
        r.status == StageStatus.FAILED for r in results.values()
    )
    all_done = all(
        r.status in (StageStatus.COMPLETED, StageStatus.SKIPPED)
        for r in results.values()
    )

    if any_failed:
        status = "failed"
    elif all_done and results:
        status = "completed"
    else:
        status = "partial"

    training_metrics = {}
    training_config = {}
    if "training" in results:
        captured = results["training"].captured_metrics
        # Separate internal keys from public metrics
        effective_config_path = captured.pop("_effective_config_path", None)
        training_metrics = captured

        # Load the effective (resolved) trainer config for index_entry lookups
        if effective_config_path:
            try:
                import yaml as _yaml
                with open(effective_config_path, "r") as f:
                    training_config = _yaml.safe_load(f) or {}
            except (OSError, Exception):
                pass  # Best effort — config may not exist in dry-run or failure

    record = ExperimentRecord(
        experiment_id=experiment_id,
        name=manifest.experiment.name,
        manifest_path=manifest.manifest_path,
        fingerprint=fingerprint,
        provenance=provenance,
        contract_version=manifest.experiment.contract_version,
        training_config=training_config,
        training_metrics=training_metrics,
        tags=manifest.experiment.tags,
        hypothesis=manifest.experiment.hypothesis,
        description=manifest.experiment.description,
        created_at=now.isoformat(),
        duration_seconds=total_duration,
        status=status,
        stages_completed=stages_completed,
    )

    ledger = ExperimentLedger(paths.ledger_dir)
    ledger.register(record)

    console.print(f"[green]Registered: {experiment_id}[/green]")
    console.print(f"  Status: {status}")
    console.print(f"  Duration: {total_duration:.1f}s")
    console.print(f"  Stages: {', '.join(stages_completed)}")


@main.command()
@click.argument("manifest_path", type=click.Path(exists=True))
@click.pass_context
def validate(ctx: click.Context, manifest_path: str) -> None:
    """Validate a manifest without running any stages."""
    pipeline_root = _resolve_pipeline_root(ctx.obj.get("pipeline_root"))
    paths = PipelinePaths(pipeline_root=pipeline_root)

    manifest = load_manifest(manifest_path)
    result = validate_manifest(manifest, paths)

    for issue in result.issues:
        color = "red" if issue.severity == "error" else "yellow"
        console.print(f"[{color}]{issue}[/{color}]")

    if result.is_valid:
        console.print("[green bold]Validation passed.[/green bold]")

        fingerprint = compute_fingerprint(manifest, paths)
        console.print(f"Fingerprint: {fingerprint[:16]}...")

        existing = check_duplicate(fingerprint, paths.ledger_dir)
        if existing:
            console.print(
                f"[yellow]Duplicate found: {existing.get('experiment_id')}[/yellow]"
            )
    else:
        console.print(
            f"[red bold]Validation failed: {len(result.errors)} errors[/red bold]"
        )
        sys.exit(1)


@main.command()
@click.option("--metric", "-m", default="training_metrics.macro_f1", help="Metric to compare.")
@click.option("--sort", "-s", default="desc", type=click.Choice(["asc", "desc"]))
@click.option("--top", "-k", type=int, default=None, help="Show top K experiments.")
@click.option("--filter-tags", type=str, default=None, help="Comma-separated tags.")
@click.option("--group-by", type=str, default=None, help="Group by field.")
@click.pass_context
def compare(
    ctx: click.Context,
    metric: str,
    sort: str,
    top: Optional[int],
    filter_tags: Optional[str],
    group_by: Optional[str],
) -> None:
    """Compare experiments by metrics."""
    pipeline_root = _resolve_pipeline_root(ctx.obj.get("pipeline_root"))
    paths = PipelinePaths(pipeline_root=pipeline_root)
    ledger = ExperimentLedger(paths.ledger_dir)

    tags = filter_tags.split(",") if filter_tags else None
    entries = ledger.filter(tags=tags) if tags else ledger.list_all()

    if not entries:
        console.print("[yellow]No experiments found in ledger.[/yellow]")
        return

    rows = compare_experiments(
        entries,
        metric_keys=[metric],
        sort_by=metric,
        ascending=(sort == "asc"),
        top_k=top,
        group_by=group_by,
    )

    metric_short = metric.split(".")[-1]
    table = Table(title=f"Experiment Comparison (sorted by {metric_short})")
    table.add_column("Name", style="cyan")
    table.add_column("Model", style="blue")
    table.add_column(metric_short, justify="right", style="green")
    table.add_column("Status")
    table.add_column("Date")

    for row in rows:
        val = row.get(metric_short, "")
        if isinstance(val, float):
            val = f"{val:.4f}"
        table.add_row(
            row.get("name", ""),
            row.get("model_type", ""),
            str(val),
            row.get("status", ""),
            row.get("created_at", ""),
        )

    console.print(table)


@main.command()
@click.argument("id_a")
@click.argument("id_b")
@click.pass_context
def diff(ctx: click.Context, id_a: str, id_b: str) -> None:
    """Show detailed diff between two experiments."""
    pipeline_root = _resolve_pipeline_root(ctx.obj.get("pipeline_root"))
    paths = PipelinePaths(pipeline_root=pipeline_root)
    ledger = ExperimentLedger(paths.ledger_dir)

    record_a = ledger.get(id_a)
    record_b = ledger.get(id_b)

    if record_a is None:
        console.print(f"[red]Experiment not found: {id_a}[/red]")
        sys.exit(1)
    if record_b is None:
        console.print(f"[red]Experiment not found: {id_b}[/red]")
        sys.exit(1)

    diff_result = diff_experiments(record_a, record_b)

    console.print(f"\n[bold]Config Differences[/bold]")
    if diff_result["config_diffs"]:
        table = Table()
        table.add_column("Key")
        table.add_column(f"A ({id_a[:20]})")
        table.add_column(f"B ({id_b[:20]})")
        for key, va, vb in diff_result["config_diffs"]:
            table.add_row(key, str(va), str(vb))
        console.print(table)
    else:
        console.print("  [dim]No config differences[/dim]")

    console.print(f"\n[bold]Metric Differences[/bold]")
    if diff_result["metric_diffs"]:
        table = Table()
        table.add_column("Metric")
        table.add_column("A", justify="right")
        table.add_column("B", justify="right")
        table.add_column("Delta", justify="right")
        for key, va, vb, delta in diff_result["metric_diffs"]:
            delta_str = ""
            if delta is not None:
                sign = "+" if delta > 0 else ""
                delta_str = f"{sign}{delta:.4f}"
            table.add_row(key, str(va or ""), str(vb or ""), delta_str)
        console.print(table)
    else:
        console.print("  [dim]No metric differences[/dim]")


@main.group()
def ledger() -> None:
    """Browse and manage the experiment ledger."""
    pass


@ledger.command(name="list")
@click.option("--status", type=str, default=None, help="Filter by status.")
@click.option("--model-type", type=str, default=None, help="Filter by model type.")
@click.pass_context
def ledger_list(
    ctx: click.Context,
    status: Optional[str],
    model_type: Optional[str],
) -> None:
    """List all experiments in the ledger."""
    pipeline_root = _resolve_pipeline_root(ctx.obj.get("pipeline_root"))
    paths = PipelinePaths(pipeline_root=pipeline_root)
    exp_ledger = ExperimentLedger(paths.ledger_dir)

    entries = exp_ledger.filter(status=status, model_type=model_type)

    if not entries:
        console.print("[yellow]No experiments found.[/yellow]")
        return

    table = Table(title=f"Experiments ({len(entries)} total)")
    table.add_column("ID", style="cyan", max_width=40)
    table.add_column("Status")
    table.add_column("Model")
    table.add_column("F1", justify="right")
    table.add_column("Acc", justify="right")
    table.add_column("Date")

    for entry in entries:
        tm = entry.get("training_metrics", {})
        f1 = tm.get("macro_f1", "")
        acc = tm.get("accuracy", "")
        table.add_row(
            entry.get("experiment_id", "")[:40],
            entry.get("status", ""),
            entry.get("model_type", ""),
            f"{f1:.4f}" if isinstance(f1, float) else str(f1),
            f"{acc:.4f}" if isinstance(acc, float) else str(acc),
            entry.get("created_at", "")[:10],
        )

    console.print(table)


@ledger.command(name="show")
@click.argument("experiment_id")
@click.pass_context
def ledger_show(ctx: click.Context, experiment_id: str) -> None:
    """Show full details of an experiment."""
    pipeline_root = _resolve_pipeline_root(ctx.obj.get("pipeline_root"))
    paths = PipelinePaths(pipeline_root=pipeline_root)
    exp_ledger = ExperimentLedger(paths.ledger_dir)

    record = exp_ledger.get(experiment_id)
    if record is None:
        console.print(f"[red]Experiment not found: {experiment_id}[/red]")
        sys.exit(1)

    console.print(f"\n[bold cyan]Experiment: {record.name}[/bold cyan]")
    console.print(f"  ID: {record.experiment_id}")
    console.print(f"  Status: {record.status}")
    console.print(f"  Created: {record.created_at}")
    console.print(f"  Duration: {record.duration_seconds:.1f}s")
    console.print(f"  Fingerprint: {record.fingerprint[:16]}...")
    console.print(f"  Contract: {record.contract_version}")

    if record.hypothesis:
        console.print(f"  Hypothesis: {record.hypothesis}")
    if record.tags:
        console.print(f"  Tags: {', '.join(record.tags)}")
    if record.notes:
        console.print(f"  Notes: {record.notes}")

    console.print(f"\n[bold]Provenance[/bold]")
    git = record.provenance.git
    console.print(f"  Git: {git.short_hash} ({git.branch}) {'[dirty]' if git.dirty else ''}")

    if record.training_metrics:
        console.print(f"\n[bold]Training Metrics[/bold]")
        for k, v in sorted(record.training_metrics.items()):
            if isinstance(v, float):
                console.print(f"  {k}: {v:.4f}")
            else:
                console.print(f"  {k}: {v}")

    if record.backtest_metrics:
        console.print(f"\n[bold]Backtest Metrics[/bold]")
        for k, v in sorted(record.backtest_metrics.items()):
            if isinstance(v, float):
                console.print(f"  {k}: {v:.4f}")
            else:
                console.print(f"  {k}: {v}")

    console.print(f"\n  Stages completed: {', '.join(record.stages_completed)}")


@ledger.command(name="search")
@click.option("--tags", type=str, default=None, help="Comma-separated tags.")
@click.option("--min-f1", type=float, default=None, help="Minimum macro F1.")
@click.option("--min-accuracy", type=float, default=None, help="Minimum accuracy.")
@click.option("--model-type", type=str, default=None, help="Model type filter.")
@click.pass_context
def ledger_search(
    ctx: click.Context,
    tags: Optional[str],
    min_f1: Optional[float],
    min_accuracy: Optional[float],
    model_type: Optional[str],
) -> None:
    """Search experiments by criteria."""
    pipeline_root = _resolve_pipeline_root(ctx.obj.get("pipeline_root"))
    paths = PipelinePaths(pipeline_root=pipeline_root)
    exp_ledger = ExperimentLedger(paths.ledger_dir)

    tag_list = tags.split(",") if tags else None
    entries = exp_ledger.filter(
        tags=tag_list,
        min_f1=min_f1,
        min_accuracy=min_accuracy,
        model_type=model_type,
    )

    console.print(f"[bold]Found {len(entries)} matching experiments[/bold]")
    for entry in entries:
        tm = entry.get("training_metrics", {})
        console.print(
            f"  {entry.get('experiment_id', '')[:40]} | "
            f"F1={tm.get('macro_f1', 'N/A')} | "
            f"Acc={tm.get('accuracy', 'N/A')} | "
            f"{entry.get('status', '')}"
        )


# =============================================================================
# Sweep Commands (Phase 4)
# =============================================================================


@main.group()
def sweep() -> None:
    """Manage parameter sweep / grid search experiments."""
    pass


@sweep.command(name="expand")
@click.argument("manifest_path", type=click.Path(exists=True))
def sweep_expand(manifest_path: str) -> None:
    """Show the grid expansion without executing (dry expansion).

    Prints all grid points with their names and effective overrides.
    """
    from hft_ops.manifest.sweep import expand_sweep, validate_sweep

    manifest = load_manifest(manifest_path)

    if manifest.sweep is None:
        console.print("[red]This manifest has no sweep section.[/red]")
        sys.exit(1)

    errors = validate_sweep(manifest.sweep)
    if errors:
        console.print("[red bold]Sweep validation failed:[/red bold]")
        for e in errors:
            console.print(f"  [red]- {e}[/red]")
        sys.exit(1)

    experiments = expand_sweep(manifest)

    table = Table(title=f"Sweep: {manifest.sweep.name} ({len(experiments)} grid points)")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Experiment Name", style="cyan")
    table.add_column("Overrides")
    table.add_column("Horizon", justify="right")

    for i, exp in enumerate(experiments, 1):
        override_str = ", ".join(
            f"{k}={v}" for k, v in exp.stages.training.overrides.items()
            if k != "data.data_dir"  # Suppress base override for readability
        )
        horizon = str(exp.stages.training.horizon_value or "")
        table.add_row(str(i), exp.experiment.name, override_str, horizon)

    console.print(table)


@sweep.command(name="run")
@click.argument("manifest_path", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Validate without executing stages.")
@click.option(
    "--stages",
    type=str,
    default=None,
    help="Comma-separated stages to run (e.g., training).",
)
@click.option("--force", is_flag=True, help="Skip duplicate checks.")
@click.option(
    "--continue-on-failure",
    is_flag=True,
    help="Continue to next grid point if one fails.",
)
@click.pass_context
def sweep_run(
    ctx: click.Context,
    manifest_path: str,
    dry_run: bool,
    stages: Optional[str],
    force: bool,
    continue_on_failure: bool,
) -> None:
    """Execute all grid points in a sweep manifest.

    Each grid point runs the same pipeline as 'hft-ops run' but with
    axis-specific overrides. Results are linked by a shared sweep_id.
    """
    from hft_ops.manifest.sweep import expand_sweep

    pipeline_root = _resolve_pipeline_root(ctx.obj.get("pipeline_root"))
    paths = PipelinePaths(pipeline_root=pipeline_root)
    ops_config = OpsConfig(
        paths=paths,
        verbose=ctx.obj.get("verbose", False),
        dry_run=dry_run,
    )

    console.print(f"[bold]Loading sweep manifest:[/bold] {manifest_path}")
    manifest = load_manifest(manifest_path)

    if manifest.sweep is None:
        console.print("[red]This manifest has no sweep section. Use 'hft-ops run' instead.[/red]")
        sys.exit(1)

    experiments = expand_sweep(manifest)
    sweep_name = manifest.sweep.name
    sweep_id = f"{sweep_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"

    console.print(
        f"[bold green]Sweep:[/bold green] {sweep_name} "
        f"({len(experiments)} grid points)"
    )
    console.print(f"[dim]Sweep ID: {sweep_id}[/dim]")

    # Validate ALL experiments before running any
    console.print(f"\n[bold]Validating all {len(experiments)} experiments...[/bold]")
    all_valid = True
    for i, exp in enumerate(experiments, 1):
        validation = validate_manifest(exp, paths)
        if not validation.is_valid:
            console.print(f"  [red]#{i} {exp.experiment.name}: INVALID[/red]")
            for issue in validation.errors:
                console.print(f"    [red]{issue}[/red]")
            all_valid = False
        else:
            console.print(f"  [green]#{i} {exp.experiment.name}: OK[/green]")

    if not all_valid:
        console.print("[red bold]Sweep validation failed. Fix errors before running.[/red bold]")
        sys.exit(1)

    if dry_run:
        console.print(f"\n[yellow bold]Dry run complete. {len(experiments)} experiments validated.[/yellow bold]")
        return

    # Execute each grid point
    requested_stages = set(stages.split(",")) if stages else None
    completed = 0
    failed = 0

    for i, exp in enumerate(experiments, 1):
        console.print(
            f"\n[bold cyan]{'='*60}[/bold cyan]"
            f"\n[bold cyan]Grid point {i}/{len(experiments)}: "
            f"{exp.experiment.name}[/bold cyan]"
        )

        # Compute fingerprint for this specific experiment
        fingerprint = compute_fingerprint(exp, paths)

        if not force:
            existing = check_duplicate(fingerprint, paths.ledger_dir)
            if existing:
                console.print(
                    f"  [yellow]Duplicate found: {existing.get('experiment_id')}. Skipping.[/yellow]"
                )
                continue

        # Build stage runners (same as `run` command)
        stage_runners = [
            ("extraction", exp.stages.extraction.enabled, ExtractionRunner()),
            ("raw_analysis", exp.stages.raw_analysis.enabled, RawAnalysisRunner()),
            ("dataset_analysis", exp.stages.dataset_analysis.enabled, DatasetAnalysisRunner()),
            ("training", exp.stages.training.enabled, TrainingRunner()),
            ("backtesting", exp.stages.backtesting.enabled, BacktestRunner()),
        ]

        results: dict[str, StageResult] = {}
        point_start = time.monotonic()
        point_failed = False

        for stage_name, enabled, runner in stage_runners:
            if not enabled:
                continue
            if requested_stages and stage_name not in requested_stages:
                continue

            input_errors = runner.validate_inputs(exp, ops_config)
            if input_errors:
                for err in input_errors:
                    console.print(f"  [red]{err}[/red]")
                results[stage_name] = StageResult(
                    stage_name=stage_name,
                    status=StageStatus.FAILED,
                    error_message="; ".join(input_errors),
                )
                point_failed = True
                break

            result = runner.run(exp, ops_config)
            results[stage_name] = result

            if result.status == StageStatus.FAILED:
                console.print(f"  [red]{stage_name} FAILED: {result.error_message}[/red]")
                point_failed = True
                break

        point_duration = time.monotonic() - point_start

        # Determine axis_values for this point
        axis_values = {}
        if manifest.sweep:
            for axis in manifest.sweep.axes:
                for val in axis.values:
                    # Check if this value's overrides are in the experiment's overrides
                    for k, v in val.overrides.items():
                        if exp.stages.training.overrides.get(k) == v or (
                            k == "horizon_value" and exp.stages.training.horizon_value == v
                        ):
                            axis_values[axis.name] = val.label
                            break

        # Record this grid point
        _record_experiment(exp, paths, fingerprint, results, point_duration)

        # Update the record with sweep metadata
        ledger = ExperimentLedger(paths.ledger_dir)
        record_id = ledger.list_ids()[-1]  # Most recently registered
        record = ledger.get(record_id)
        if record:
            record.sweep_id = sweep_id
            record.axis_values = axis_values
            record.save(paths.ledger_dir / "records" / f"{record_id}.json")
            # Rebuild index with updated sweep fields
            ledger._rebuild_index()

        if point_failed:
            failed += 1
            if not continue_on_failure:
                console.print(
                    f"[red bold]Sweep aborted at grid point {i}. "
                    f"Use --continue-on-failure to proceed.[/red bold]"
                )
                break
        else:
            completed += 1

    # Summary
    console.print(f"\n[bold]{'='*60}[/bold]")
    console.print(f"[bold]Sweep complete: {sweep_name}[/bold]")
    console.print(f"  Completed: [green]{completed}[/green]")
    console.print(f"  Failed: [red]{failed}[/red]")
    console.print(f"  Skipped (dupes): {len(experiments) - completed - failed}")
    console.print(f"  Sweep ID: {sweep_id}")


@sweep.command(name="results")
@click.argument("sweep_id")
@click.option(
    "--metric", "-m",
    type=str,
    default="training_metrics.best_val_macro_f1",
    help="Metric to sort by.",
)
@click.option("--sort", "-s", type=click.Choice(["asc", "desc"]), default="desc")
@click.pass_context
def sweep_results(
    ctx: click.Context,
    sweep_id: str,
    metric: str,
    sort: str,
) -> None:
    """Show comparison of all experiments in a sweep."""
    pipeline_root = _resolve_pipeline_root(ctx.obj.get("pipeline_root"))
    paths = PipelinePaths(pipeline_root=pipeline_root)
    ledger = ExperimentLedger(paths.ledger_dir)

    entries = ledger.filter(sweep_id=sweep_id)

    if not entries:
        console.print(f"[yellow]No experiments found for sweep_id: {sweep_id}[/yellow]")
        return

    # Build comparison table
    rows = compare_experiments(
        entries,
        metric_keys=[metric],
        sort_by=metric.split(".")[-1],
        ascending=(sort == "asc"),
    )

    table = Table(title=f"Sweep Results: {sweep_id} ({len(rows)} points)")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Experiment", style="cyan")
    table.add_column("Axes")
    table.add_column(metric.split(".")[-1], justify="right")
    table.add_column("Status")

    for i, row in enumerate(rows, 1):
        # Get axis values from the index entry
        entry = next(
            (e for e in entries if e["experiment_id"] == row.get("experiment_id")),
            {},
        )
        axis_str = ", ".join(
            f"{k}={v}" for k, v in entry.get("axis_values", {}).items()
        )
        metric_val = row.get(metric.split(".")[-1], "")
        metric_str = f"{metric_val:.4f}" if isinstance(metric_val, float) else str(metric_val)

        table.add_row(
            str(i),
            row.get("name", ""),
            axis_str,
            metric_str,
            row.get("status", ""),
        )

    console.print(table)


@main.command(name="check-dup")
@click.argument("manifest_path", type=click.Path(exists=True))
@click.pass_context
def check_dup(ctx: click.Context, manifest_path: str) -> None:
    """Check if an experiment would be a duplicate."""
    pipeline_root = _resolve_pipeline_root(ctx.obj.get("pipeline_root"))
    paths = PipelinePaths(pipeline_root=pipeline_root)

    manifest = load_manifest(manifest_path)
    fingerprint = compute_fingerprint(manifest, paths)

    console.print(f"Fingerprint: {fingerprint[:16]}...")

    existing = check_duplicate(fingerprint, paths.ledger_dir)
    if existing:
        console.print(
            f"[yellow bold]Duplicate found:[/yellow bold] "
            f"{existing.get('experiment_id')}"
        )
        console.print(f"  Created: {existing.get('created_at', '')}")
        console.print(f"  Status: {existing.get('status', '')}")
    else:
        console.print("[green]No duplicate found. Safe to run.[/green]")


if __name__ == "__main__":
    main()
