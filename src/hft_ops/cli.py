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

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import click
from rich.console import Console
from rich.table import Table

from hft_ops.config import OpsConfig
from hft_ops.ledger.comparator import compare_experiments, diff_experiments
from hft_ops.ledger.dedup import check_duplicate, compute_fingerprint
from hft_ops.ledger.experiment_record import ExperimentRecord
from hft_ops.ledger.ledger import ExperimentLedger
from hft_ops.manifest.loader import load_manifest
from hft_ops.manifest.validator import (
    apply_resolved_context,
    resolve_manifest_context,
    validate_manifest,
)
from hft_ops.paths import PipelinePaths
from hft_ops.provenance.lineage import build_provenance
from hft_ops.stages.backtesting import BacktestRunner
from hft_ops.stages.post_training_gate import PostTrainingGateRunner
from hft_ops.stages.signal_export import SignalExportRunner
from hft_ops.stages.base import StageResult, StageStatus
from hft_ops.stages.dataset_analysis import DatasetAnalysisRunner
from hft_ops.stages.extraction import ExtractionRunner
from hft_ops.stages.raw_analysis import RawAnalysisRunner
from hft_ops.stages.training import TrainingRunner
from hft_ops.stages.validation import ValidationRunner

console = Console()


def _summarize_training_history(history: list) -> dict:
    """Summarize a per-epoch training history list into a flat metrics dict.

    Detects "best epoch" by minimizing ``val_loss`` (fallback: max ``val_macro_f1``
    if val_loss absent) and exposes that epoch's metrics with ``best_`` prefix,
    plus the final epoch's metrics with ``final_`` prefix.

    This lets historical runs (which stored per-epoch trajectories) surface
    the same summary fields that new hft-ops-orchestrated runs capture.
    """
    if not history:
        return {}

    # Determine best-epoch key — prefer val_loss (minimize), fallback val_macro_f1 (maximize)
    has_val_loss = any("val_loss" in h for h in history if isinstance(h, dict))
    has_val_f1 = any("val_macro_f1" in h for h in history if isinstance(h, dict))

    if has_val_loss:
        best = min(
            (h for h in history if isinstance(h, dict) and "val_loss" in h),
            key=lambda h: h["val_loss"],
        )
    elif has_val_f1:
        best = max(
            (h for h in history if isinstance(h, dict) and "val_macro_f1" in h),
            key=lambda h: h["val_macro_f1"],
        )
    else:
        best = history[-1] if isinstance(history[-1], dict) else {}

    final = history[-1] if isinstance(history[-1], dict) else {}

    summary: dict = {"n_epochs": len(history)}

    for k, v in best.items():
        if k == "epoch":
            summary["best_epoch"] = v
        elif k.startswith("val_") or k == "train_loss":
            summary[f"best_{k}"] = v

    for k, v in final.items():
        if k == "epoch":
            summary["final_epoch"] = v
        elif k.startswith("val_") or k == "train_loss":
            summary[f"final_{k}"] = v

    # Provide common aliases the ledger-list view expects
    if "best_val_macro_f1" in summary:
        summary["macro_f1"] = summary["best_val_macro_f1"]
    if "best_val_accuracy" in summary:
        summary["accuracy"] = summary["best_val_accuracy"]
    if "best_val_ic" in summary:
        summary["ic"] = summary["best_val_ic"]
    if "best_val_r2" in summary:
        summary["r_squared"] = summary["best_val_r2"]

    return summary


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

    # Resolve runtime values (horizon_idx, feature_count) once and apply them
    # explicitly to the manifest. validate_manifest is now side-effect free —
    # this is the single orchestrator-level mutation point.
    resolved_ctx = resolve_manifest_context(manifest, paths)
    apply_resolved_context(manifest, resolved_ctx)

    requested_stages = None
    if stages:
        requested_stages = set(stages.split(","))

    # Stage order mirrors the schema's pipeline: extraction → raw/dataset
    # analysis → validation (Rule-13 IC gate) → training → signal_export →
    # backtesting. Order matters: validation MUST run before training so a
    # failing gate under on_fail=abort prevents wasted training compute.
    stage_runners = [
        ("extraction", manifest.stages.extraction.enabled, ExtractionRunner()),
        ("raw_analysis", manifest.stages.raw_analysis.enabled, RawAnalysisRunner()),
        ("dataset_analysis", manifest.stages.dataset_analysis.enabled, DatasetAnalysisRunner()),
        ("validation", manifest.stages.validation.enabled, ValidationRunner()),
        ("training", manifest.stages.training.enabled, TrainingRunner()),
        # Phase 7 Stage 7.4 (2026-04-19): post-training regression-detection
        # gate. Runs between training and signal_export so a researcher gets
        # quality signal BEFORE compute-intensive signal export + backtest.
        # Default enabled=False; opt-in via manifest.
        (
            "post_training_gate",
            manifest.stages.post_training_gate.enabled,
            PostTrainingGateRunner(),
        ),
        ("signal_export", manifest.stages.signal_export.enabled, SignalExportRunner()),
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
) -> str:
    """Build and store an ExperimentRecord from stage results.

    Returns:
        The `experiment_id` of the just-registered record. Phase 6 6A.11
        added this return value so sweep-loop callers can reference the
        freshly-registered record directly instead of the brittle
        `ledger.list_ids()[-1]` pattern (which couples correctness to
        append-order of the in-memory index across `ExperimentLedger(...)`
        re-instantiations).
    """
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%S")
    experiment_id = f"{manifest.experiment.name}_{timestamp}_{fingerprint[:8]}"

    ext_config_path = None
    if manifest.stages.extraction.config:
        ext_config_path = paths.resolve(manifest.stages.extraction.config)

    # Phase 6 6A.3 (2026-04-17, revised after validation audit): dispatch to
    # file-path OR inline-dict hashing via `build_provenance`'s symmetric API
    # (mutually exclusive per lineage.py contract). Inline `trainer_config:`
    # (Phase 1 wrapper-less) paths produce `config_hashes["trainer"]` via
    # canonical_hash SSoT; file paths produce it via hash_file. Removed the
    # prior post-mutation pattern (fragile; blocked Phase 6B.4 Provenance →
    # hft_contracts migration because hft_contracts can't import from hft-ops).
    train_config_path = None
    train_config_dict = None
    if manifest.stages.training.config:
        train_config_path = paths.resolve(manifest.stages.training.config)
    elif manifest.stages.training.trainer_config is not None:
        train_config_dict = manifest.stages.training.trainer_config

    provenance = build_provenance(
        paths.pipeline_root,
        manifest_path=Path(manifest.manifest_path) if manifest.manifest_path else None,
        extractor_config_path=ext_config_path,
        trainer_config_path=train_config_path,
        trainer_config_dict=train_config_dict,
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

    # Phase 7 Stage 7.4 post-validation (2026-04-19): B-H1 fix — harvest the
    # post_training_gate report from the gate's captured_metrics and
    # persist it IN the ExperimentRecord so researchers can:
    #   (a) see which prior experiments hit regression warnings
    #       (`hft-ops ledger show <id>` shows full record including gate)
    #   (b) write ad-hoc queries like "experiments with gate status=warn"
    #       via `hft-ops ledger list` or programmatic filter on
    #       `training_metrics.post_training_gate.status`
    # Pre-fix: gate wrote gate_report.json to disk but NEVER reached the
    # record — ephemeral only, invisible to all ledger tooling.
    # Implementation: nest the report under training_metrics (a
    # Dict[str, Any] already typed for free-form sub-dicts; no schema
    # change required on ExperimentRecord). Includes both the full
    # report dict AND the one-line summary for grep-friendly display.
    if "post_training_gate" in results:
        gate_captured = results["post_training_gate"].captured_metrics
        if "post_training_gate" in gate_captured:
            training_metrics["post_training_gate"] = gate_captured[
                "post_training_gate"
            ]
        if "post_training_gate_summary" in gate_captured:
            training_metrics["post_training_gate_summary"] = gate_captured[
                "post_training_gate_summary"
            ]

    # Phase 4 Batch 4c.4 (2026-04-16): harvest `feature_set_ref` from
    # signal_export's captured_metrics (populated by
    # `SignalExportRunner._harvest_feature_set_ref` from signal_metadata.json).
    # None iff signal_export stage was skipped/failed OR the trainer did not
    # use DataConfig.feature_set. ExperimentRecord stores None gracefully.
    feature_set_ref: Optional[Dict[str, str]] = None
    if "signal_export" in results:
        raw_ref = results["signal_export"].captured_metrics.get("feature_set_ref")
        if isinstance(raw_ref, dict):
            name = raw_ref.get("name")
            content_hash = raw_ref.get("content_hash")
            if isinstance(name, str) and isinstance(content_hash, str):
                feature_set_ref = {"name": name, "content_hash": content_hash}

    record = ExperimentRecord(
        experiment_id=experiment_id,
        name=manifest.experiment.name,
        manifest_path=manifest.manifest_path,
        fingerprint=fingerprint,
        feature_set_ref=feature_set_ref,
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
    return experiment_id


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

    # Phase 5 FULL-A post-audit fix (Agent 4 Issue 3): exclude sweep-aggregate
    # parent records from the cross-experiment comparison table — they have no
    # training_metrics/backtest_metrics of their own (children carry those).
    # Dashboards that want the aggregate can opt in by filtering
    # record_type="sweep_aggregate" separately.
    entries = [e for e in entries if e.get("record_type") != "sweep_aggregate"]

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


@ledger.command(name="rebuild-index")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be rebuilt without writing index.json.",
)
@click.pass_context
def ledger_rebuild_index(ctx: click.Context, dry_run: bool) -> None:
    """Rebuild index.json from individual record files.

    Phase 7 Stage 7.4 post-validation (2026-04-19) C2 fix. When the
    ``ExperimentRecord.index_entry()`` whitelist is expanded (e.g., the
    Phase 7.4 addition of 7 regression metric keys), the cached
    ``index.json`` retains the OLD projection for every historical
    record. The post-training gate's ``_find_prior_best_experiment``
    iterates this index — so until index.json is rebuilt, historical
    experiments appear to lack the new metrics entirely, defeating the
    gate's primary purpose.

    This subcommand re-projects every record under
    ``ledger_dir/records/*.json`` via the current ``index_entry()``
    whitelist and overwrites ``index.json``. Idempotent (running twice
    produces identical output modulo record-ordering).

    Usage:

        hft-ops ledger rebuild-index           # rebuild + report
        hft-ops ledger rebuild-index --dry-run # count-only, no write
    """
    pipeline_root = _resolve_pipeline_root(ctx.obj.get("pipeline_root"))
    paths = PipelinePaths(pipeline_root=pipeline_root)
    ledger_dir = paths.ledger_dir
    records_dir = ledger_dir / "records"

    if not records_dir.exists():
        console.print(f"[yellow]No records dir at {records_dir}; nothing to rebuild.[/yellow]")
        return

    n_records_on_disk = len(list(records_dir.glob("*.json")))

    # Open existing ledger to capture the pre-rebuild index size.
    ledger = ExperimentLedger(ledger_dir)
    old_index_len = len(ledger._index)

    if dry_run:
        console.print(
            f"[yellow]DRY RUN[/yellow]: would rebuild from "
            f"[cyan]{n_records_on_disk}[/cyan] record files."
        )
        console.print(f"  Current index.json: [dim]{old_index_len} entries[/dim]")
        console.print(
            f"  After rebuild:      [dim]~{n_records_on_disk} entries "
            f"(minus any malformed)[/dim]"
        )
        return

    console.print(
        f"[cyan]Rebuilding index from {n_records_on_disk} record files…[/cyan]"
    )
    ledger._rebuild_index()
    new_index_len = len(ledger._index)

    console.print(
        f"[green]Index rebuilt: {old_index_len} → {new_index_len} entries[/green]"
    )
    if new_index_len != old_index_len:
        delta = new_index_len - old_index_len
        console.print(
            f"[yellow]  Delta: {delta:+d} "
            f"({'new entries detected' if delta > 0 else 'malformed records skipped'})[/yellow]"
        )
    skipped = n_records_on_disk - new_index_len
    if skipped > 0:
        console.print(
            f"[yellow]  Note: {skipped} record file(s) could not be "
            f"re-projected (malformed JSON or schema mismatch).[/yellow]"
        )


@ledger.command(name="fingerprint-explain")
@click.argument("manifest_path", type=click.Path(exists=True))
@click.option(
    "--indent",
    type=int,
    default=2,
    show_default=True,
    help="JSON indent for the explained-components dump.",
)
@click.pass_context
def ledger_fingerprint_explain(
    ctx: click.Context,
    manifest_path: str,
    indent: int,
) -> None:
    """Dump the normalized `components` dict that would be hashed for a manifest.

    Phase 4 Batch 4c.3 Enhancement A: self-service debugging for "why do these
    two manifests fingerprint differently?" + Phase 10 parity audits.

    Prints the FULL components dict as pretty JSON to stderr, then the computed
    fingerprint hex to stdout. Two manifests that normalize identically will
    produce byte-equal stderr output. Redirect stderr to files and diff them.

    Usage:
        hft-ops ledger fingerprint-explain manifest_a.yaml 2> a.json
        hft-ops ledger fingerprint-explain manifest_b.yaml 2> b.json
        diff a.json b.json
    """
    import sys as _sys
    from hft_ops.manifest.loader import load_manifest
    from hft_ops.ledger.dedup import compute_fingerprint_explain

    pipeline_root = _resolve_pipeline_root(ctx.obj.get("pipeline_root"))
    paths = PipelinePaths(pipeline_root=pipeline_root)

    manifest = load_manifest(Path(manifest_path))
    fp, components = compute_fingerprint_explain(manifest, paths)

    # JSON-dump components to stderr (audit target); fingerprint to stdout
    # (machine-parseable for diff tools / CI).
    _sys.stderr.write(json.dumps(components, sort_keys=True, indent=indent, default=str))
    _sys.stderr.write("\n")
    console.print(f"fingerprint: {fp}")


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


@ledger.command(name="backfill")
@click.argument("manifest_path", type=click.Path(exists=True))
@click.option(
    "--metrics-file",
    type=click.Path(exists=True),
    required=False,
    default=None,
    help="Path to existing training_history.json / results.json / classification_table.json. "
         "Optional for analysis-only / cancelled experiments.",
)
@click.option(
    "--record-type",
    type=click.Choice([
        "training", "analysis", "calibration", "backtest", "evaluation", "sweep_aggregate",
    ]),
    default="training",
    help="Type of record. See RecordType enum docstring for definitions.",
)
@click.option(
    "--parent-id",
    type=str,
    default="",
    help="Parent experiment ID (required for calibration / dependent backtest types).",
)
@click.option(
    "--status",
    type=click.Choice(["completed", "failed", "cancelled", "partial"]),
    default="completed",
    help="Final status of the historical experiment.",
)
@click.option(
    "--notes",
    type=str,
    default="",
    help="Free-form notes about this retroactive record (e.g., gaps in artifacts).",
)
@click.pass_context
def ledger_backfill(
    ctx: click.Context,
    manifest_path: str,
    metrics_file: Optional[str],
    record_type: str,
    parent_id: str,
    status: str,
    notes: str,
) -> None:
    """Backfill a ledger record from a historical experiment.

    Used to populate the ledger retroactively for E1-E16-style experiments
    that were run before hft-ops became the orchestrator. Marks the record
    with provenance.retroactive=True; uses the NOT_GIT_TRACKED_SENTINEL
    when the monorepo isn't a git repo.

    The manifest_path is a metadata-only manifest (typically under
    hft-ops/experiments/retroactive/) describing the historical experiment.
    The actual training / analysis / backtest stages should be disabled in
    that manifest — the artifacts already exist on disk.

    Examples:

        hft-ops ledger backfill experiments/retroactive/e4_tlob_h60.yaml \\
            --metrics-file ../lob-model-trainer/outputs/experiments/e4_tlob_h60/training_history.json \\
            --record-type training

        hft-ops ledger backfill experiments/retroactive/e7_regime.yaml \\
            --record-type analysis \\
            --notes "Phase A analysis-only; no training run"
    """
    import json as _json
    from datetime import datetime as _datetime, timezone as _timezone
    from hft_ops.manifest.loader import load_manifest as _load_manifest
    from hft_ops.provenance.lineage import build_provenance, NOT_GIT_TRACKED_SENTINEL

    pipeline_root = _resolve_pipeline_root(ctx.obj.get("pipeline_root"))
    paths = PipelinePaths(pipeline_root=pipeline_root)
    exp_ledger = ExperimentLedger(paths.ledger_dir)

    manifest = _load_manifest(Path(manifest_path))

    # Build provenance with retroactive marker. Use file mtime of metrics_file
    # if provided, otherwise current time. Git info uses sentinel if monorepo
    # isn't a repo.
    if metrics_file:
        ts_unix = Path(metrics_file).stat().st_mtime
        timestamp_utc = _datetime.fromtimestamp(ts_unix, tz=_timezone.utc).isoformat()
    else:
        timestamp_utc = _datetime.now(_timezone.utc).isoformat()

    prov = build_provenance(
        pipeline_root=pipeline_root,
        manifest_path=Path(manifest_path),
        contract_version=manifest.experiment.contract_version,
    )
    prov.retroactive = True
    prov.timestamp_utc = timestamp_utc

    # Load metrics if provided. Best-effort; record gets minimal metrics if missing.
    training_metrics: dict = {}
    backtest_metrics: dict = {}
    sub_records: list = []
    if metrics_file:
        try:
            with open(metrics_file, "r") as f:
                metrics_data = _json.load(f)

            if isinstance(metrics_data, list):
                # Heuristic: a list of per-epoch dicts (training history) vs
                # a list of sub-experiments (sweep aggregate).
                if metrics_data and isinstance(metrics_data[0], dict) and "epoch" in metrics_data[0]:
                    # Training history: summarize as best-epoch + final-epoch metrics.
                    training_metrics = _summarize_training_history(metrics_data)
                else:
                    # Sub-records for sweep_aggregate
                    sub_records = metrics_data
            elif "sub_runs" in metrics_data:
                sub_records = metrics_data["sub_runs"]
            else:
                # Common training_history.json shape: {"train": [...], "val": [...]}
                # OR a flat metrics dict — keep the whole thing
                training_metrics = metrics_data.get("metrics", metrics_data)
                if "backtest_metrics" in metrics_data:
                    backtest_metrics = metrics_data["backtest_metrics"]
        except (json.JSONDecodeError, OSError) as e:
            console.print(
                f"[yellow]Warning: failed to load metrics_file ({e}); "
                f"creating record with empty metrics[/yellow]"
            )

    # Compute fingerprint of the manifest config (no actual run state)
    fingerprint = compute_fingerprint(manifest, paths)

    # Build experiment_id
    short_ts = timestamp_utc.replace("-", "").replace(":", "").replace(".", "")[:15]
    experiment_id = f"{manifest.experiment.name}_{short_ts}_{fingerprint[:8]}_retro"

    # Check for duplicates before registering. `find_by_fingerprint` returns
    # an index-entry dict (or None), NOT a list of records.
    existing = exp_ledger.find_by_fingerprint(fingerprint)
    if existing is not None:
        existing_id = existing.get("experiment_id", "<unknown>")
        console.print(
            f"[yellow]Warning: a record with the same fingerprint already exists "
            f"({existing_id}). Skipping; record not re-registered.[/yellow]"
        )
        # Exit 0: duplicate is a BENIGN SKIP in backfill context. Matches the
        # behavior of `hft-ops run` (see line ~203) and sweep (~977). Idempotent
        # re-runs of the retroactive generator should succeed, not fail.
        sys.exit(0)

    record = ExperimentRecord(
        experiment_id=experiment_id,
        name=manifest.experiment.name,
        manifest_path=str(manifest_path),
        fingerprint=fingerprint,
        provenance=prov,
        contract_version=manifest.experiment.contract_version,
        training_metrics=training_metrics,
        backtest_metrics=backtest_metrics,
        sub_records=sub_records,
        tags=list(manifest.experiment.tags) + ["retroactive"],
        hypothesis=manifest.experiment.hypothesis,
        description=manifest.experiment.description,
        notes=notes,
        record_type=record_type,
        parent_experiment_id=parent_id,
        status=status,
        stages_completed=[],
        created_at=timestamp_utc,
    )

    exp_ledger.register(record)
    console.print(
        f"[green]✓ Backfilled record:[/green] {experiment_id}\n"
        f"  type: {record_type} | status: {status} | retroactive: True\n"
        f"  git: {prov.git.commit_hash[:16] if prov.git.commit_hash != NOT_GIT_TRACKED_SENTINEL else NOT_GIT_TRACKED_SENTINEL}"
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

    # Phase 5 Preview fix (B1): use `expand_sweep_with_axis_values` so axis
    # labels are piped through directly from the Cartesian-product expansion
    # instead of being lossily re-derived from applied overrides. The prior
    # back-derivation heuristic (deleted below) mislabeled grid points under
    # multi-axis overlap.
    from hft_ops.manifest.sweep import expand_sweep_with_axis_values
    experiments_with_axes = expand_sweep_with_axis_values(manifest)
    experiments = [exp for exp, _ in experiments_with_axes]
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
    # Phase 5 FULL-A Block 3: accumulate per-grid-point summaries for the
    # aggregate record written at loop end.
    child_summaries: list[dict] = []

    for i, (exp, axis_values) in enumerate(experiments_with_axes, 1):
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
                # Phase 5 FULL-A post-audit fix (Agent 2 B1): record the
                # skipped-duplicate grid point in child_summaries so the
                # aggregate record correctly represents sweep completeness.
                # Otherwise the aggregate sub_records count differs from the
                # expanded grid cardinality and CRITICAL-FIX 8's cross-invocation
                # aggregate_fp stability is silently violated under mixed
                # force/no-force re-runs.
                child_summaries.append({
                    "experiment_id": existing.get("experiment_id", ""),
                    "name": exp.experiment.name,
                    "fingerprint": fingerprint,
                    "axis_values": axis_values,
                    "status": "skipped_duplicate",
                    "duration_seconds": 0.0,
                    "training_metrics": {},
                    "backtest_metrics": {},
                })
                continue

        # Resolve per-grid-point runtime values (no shared-state mutation).
        resolved_ctx = resolve_manifest_context(exp, paths)
        apply_resolved_context(exp, resolved_ctx)

        # Build stage runners (same as `run` command)
        stage_runners = [
            ("extraction", exp.stages.extraction.enabled, ExtractionRunner()),
            ("raw_analysis", exp.stages.raw_analysis.enabled, RawAnalysisRunner()),
            ("dataset_analysis", exp.stages.dataset_analysis.enabled, DatasetAnalysisRunner()),
            ("validation", exp.stages.validation.enabled, ValidationRunner()),
            ("training", exp.stages.training.enabled, TrainingRunner()),
            # Phase 7 Stage 7.4 (2026-04-19): post-training regression gate
            (
                "post_training_gate",
                exp.stages.post_training_gate.enabled,
                PostTrainingGateRunner(),
            ),
            ("signal_export", exp.stages.signal_export.enabled, SignalExportRunner()),
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

        # Phase 5 Preview (B1 fix): `axis_values` is now piped directly from
        # expand_sweep_with_axis_values — ground truth from the Cartesian
        # product, not a lossy heuristic over applied overrides. Prior code
        # here had a match-wins bug under multi-axis overlap (deleted
        # 2026-04-16).

        # Record this grid point. Phase 6 6A.11 (revised after validation
        # audit): use the returned experiment_id directly instead of the
        # brittle `ledger.list_ids()[-1]` pattern. The prior pattern relied
        # on append-order preservation across `ExperimentLedger(...)`
        # re-instantiations AND was vulnerable to timestamp-collision if
        # two points register in the same second with identical experiment_name.
        record_id = _record_experiment(exp, paths, fingerprint, results, point_duration)

        # Update the record with sweep metadata
        ledger = ExperimentLedger(paths.ledger_dir)
        record = ledger.get(record_id)
        if record:
            record.sweep_id = sweep_id
            record.axis_values = axis_values
            record.save(paths.ledger_dir / "records" / f"{record_id}.json")
            # Phase 6 6A.11 REVISED (post-validation-audit 2026-04-18):
            # O(1) in-place index update. Prior Phase 6 6A.11 removed the
            # per-point _rebuild_index() (was O(N² ledger_size), correctly
            # flagged as too expensive). But merely calling record.save()
            # writes records/<id>.json — index.json entry still carries
            # sweep_id="" / axis_values={} from `register()`'s initial
            # snapshot. On SIGKILL/crash mid-sweep, index.json stays stale
            # until manual rebuild. Fix: targeted in-place update of the
            # matching index entry — O(N) scan, O(1) per-point vs O(N²)
            # rebuild, AND crash-safe because index.json reflects the
            # mutated state immediately.
            for entry in ledger._index:
                if entry.get("experiment_id") == record_id:
                    entry["sweep_id"] = sweep_id
                    entry["axis_values"] = axis_values
                    break
            ledger._save_index()

        # Phase 5 FULL-A Block 3: accumulate per-point summary for the
        # aggregate record written at loop end.
        child_summaries.append({
            "experiment_id": record_id,
            "name": exp.experiment.name,
            "fingerprint": fingerprint,
            "axis_values": axis_values,
            "status": "failed" if point_failed else "completed",
            "duration_seconds": point_duration,
            "training_metrics": (record.training_metrics if record else {}),
            "backtest_metrics": (record.backtest_metrics if record else {}),
        })

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

    # Phase 5 FULL-A Block 3: write ONE sweep_aggregate record summarizing
    # the entire sweep run. See SweepAggregateWriter for fingerprint,
    # path-placement, and lifecycle semantics.
    if child_summaries:
        from hft_ops.ledger.sweep_aggregate import SweepAggregateWriter
        SweepAggregateWriter().write(
            ledger_dir=paths.ledger_dir,
            sweep_id=sweep_id,
            sweep_name=sweep_name,
            manifest=manifest,
            child_summaries=child_summaries,
            completed=completed,
            failed=failed,
        )
        # Rebuild index so the aggregate record is visible to ledger.filter.
        ExperimentLedger(paths.ledger_dir)._rebuild_index()

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

    # Phase 5 FULL-A CRITICAL-FIX 3: exclude the sweep_aggregate parent record
    # so the results table shows grid points only. Dashboards that want to
    # surface the aggregate can opt in by filtering record_type="sweep_aggregate"
    # separately.
    entries = [e for e in entries if e.get("record_type") != "sweep_aggregate"]

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


# ===========================================================================
# Phase 4: FeatureSet Registry CLI
# ===========================================================================


@main.command(name="evaluate")
@click.option(
    "--config",
    "evaluator_config",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the hft-feature-evaluator YAML config.",
)
@click.option(
    "--criteria",
    "criteria_yaml",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the SelectionCriteria YAML.",
)
@click.option(
    "--save-feature-set",
    "save_name",
    type=str,
    required=True,
    help="FeatureSet identifier (typically '<base>_v<N>', e.g. 'momentum_hft_v1'). "
         "This becomes the <name>.json filename under contracts/feature_sets/.",
)
@click.option(
    "--applies-to-assets",
    type=str,
    required=True,
    help="Comma-separated ticker symbols (e.g. 'NVDA' or 'NVDA,MSFT').",
)
@click.option(
    "--applies-to-horizons",
    type=str,
    required=True,
    help="Comma-separated label horizons (e.g. '60' or '10,60,300').",
)
@click.option(
    "--description",
    type=str,
    default="",
    help="Free-text description (metadata, not hashed).",
)
@click.option(
    "--notes",
    type=str,
    default="",
    help="Free-text operator notes (metadata, not hashed).",
)
@click.option(
    "--created-by",
    type=str,
    default="",
    help="Producer/operator identifier.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite an existing FeatureSet with a DIFFERENT content hash. "
         "Idempotent no-ops (same content) do not require --force.",
)
@click.pass_context
def evaluate(
    ctx: click.Context,
    evaluator_config: str,
    criteria_yaml: str,
    save_name: str,
    applies_to_assets: str,
    applies_to_horizons: str,
    description: str,
    notes: str,
    created_by: str,
    force: bool,
) -> None:
    """Run the feature evaluator and persist the selected set as a FeatureSet.

    Content-addressed: the output file's content_hash is SHA-256 over
    product fields only (feature_indices + source_feature_count +
    contract_version). Identical products produce identical hashes
    regardless of criteria/name/asset differences.

    Exits:
        0 — success (new FeatureSet written OR idempotent match on existing)
        1 — evaluator not installed OR evaluator run failed
        2 — overwrite refused (different content exists, --force missing)
    """
    from hft_ops.feature_sets import (
        EvaluatorNotInstalled,
        FeatureSetExists,
        NoFeaturesSelectedError,
        produce_feature_set,
        write_feature_set,
    )

    pipeline_root = _resolve_pipeline_root(ctx.obj.get("pipeline_root"))
    paths = PipelinePaths(pipeline_root=pipeline_root)

    assets = [a.strip() for a in applies_to_assets.split(",") if a.strip()]
    if not assets:
        console.print("[red]--applies-to-assets must be non-empty[/red]")
        sys.exit(1)
    try:
        horizons = [int(h.strip()) for h in applies_to_horizons.split(",") if h.strip()]
    except ValueError as exc:
        console.print(f"[red]--applies-to-horizons must be comma-separated ints: {exc}[/red]")
        sys.exit(1)
    if not horizons:
        console.print("[red]--applies-to-horizons must be non-empty[/red]")
        sys.exit(1)

    target_path = paths.feature_sets_dir / f"{save_name}.json"
    console.print(f"[cyan]Running evaluator → {target_path}[/cyan]")

    try:
        feature_set = produce_feature_set(
            evaluator_config_path=Path(evaluator_config),
            criteria_yaml_path=Path(criteria_yaml),
            name=save_name,
            applies_to_assets=assets,
            applies_to_horizons=horizons,
            pipeline_paths=paths,
            description=description,
            notes=notes,
            created_by=created_by,
        )
    except EvaluatorNotInstalled as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)
    except NoFeaturesSelectedError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    try:
        written_path = write_feature_set(target_path, feature_set, force=force)
    except FeatureSetExists as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        sys.exit(2)

    console.print(
        f"[green]FeatureSet '{save_name}' saved.[/green]\n"
        f"  Path:         {written_path}\n"
        f"  Content hash: {feature_set.content_hash[:16]}...\n"
        f"  Features:     {len(feature_set.feature_indices)} / "
        f"{feature_set.source_feature_count}\n"
        f"  Contract:     {feature_set.contract_version}"
    )


@main.group(name="feature-sets")
def feature_sets_group() -> None:
    """Phase 4 FeatureSet registry browsing (list + show)."""


@feature_sets_group.command(name="list")
@click.pass_context
def feature_sets_list(ctx: click.Context) -> None:
    """List all FeatureSets in the registry.

    Each entry shows name + first 16 hex chars of content_hash. Does
    NOT verify integrity — call `feature-sets show <name>` for that.
    """
    from hft_ops.feature_sets import FeatureSetRegistry

    pipeline_root = _resolve_pipeline_root(ctx.obj.get("pipeline_root"))
    paths = PipelinePaths(pipeline_root=pipeline_root)

    reg = FeatureSetRegistry(paths.feature_sets_dir, allow_missing=True)
    refs = reg.list_refs()

    if not refs:
        console.print(f"[dim]No FeatureSets found at {paths.feature_sets_dir}[/dim]")
        return

    table = Table(title=f"FeatureSets ({len(refs)})")
    table.add_column("Name", style="cyan")
    table.add_column("Content hash (short)")

    for ref in refs:
        table.add_row(ref.name, f"{ref.content_hash[:16]}...")

    console.print(table)


@feature_sets_group.command(name="show")
@click.argument("name", type=str)
@click.option(
    "--no-verify",
    is_flag=True,
    help="Skip integrity verification (allow inspecting tampered files).",
)
@click.pass_context
def feature_sets_show(
    ctx: click.Context,
    name: str,
    no_verify: bool,
) -> None:
    """Show the full contents of a FeatureSet.

    Runs integrity verification by default (fails with exit code 1 on
    hash mismatch); pass --no-verify to inspect a known-bad file.
    """
    from hft_ops.feature_sets import (
        FeatureSetIntegrityError,
        FeatureSetNotFound,
        FeatureSetRegistry,
    )

    pipeline_root = _resolve_pipeline_root(ctx.obj.get("pipeline_root"))
    paths = PipelinePaths(pipeline_root=pipeline_root)

    reg = FeatureSetRegistry(paths.feature_sets_dir, allow_missing=True)

    try:
        fs = reg.get(name, verify=not no_verify)
    except FeatureSetNotFound as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)
    except FeatureSetIntegrityError as exc:
        console.print(f"[red]Integrity check failed:[/red] {exc}")
        sys.exit(1)

    console.print(f"[bold cyan]{fs.name}[/bold cyan]")
    console.print(f"  Content hash:         {fs.content_hash}")
    console.print(f"  Schema version:       {fs.schema_version}")
    console.print(f"  Contract version:     {fs.contract_version}")
    console.print(f"  Source feature count: {fs.source_feature_count}")
    console.print(
        f"  Feature indices:      {len(fs.feature_indices)} selected "
        f"({list(fs.feature_indices[:10])}{'...' if len(fs.feature_indices) > 10 else ''})"
    )
    console.print(f"  Applies-to assets:    {list(fs.applies_to.assets)}")
    console.print(f"  Applies-to horizons:  {list(fs.applies_to.horizons)}")
    console.print(f"  Produced by tool:     {fs.produced_by.tool} {fs.produced_by.tool_version}")
    console.print(f"  Source profile hash:  {fs.produced_by.source_profile_hash[:16]}...")
    console.print(f"  Criteria schema:      v{fs.criteria_schema_version}")
    console.print(f"  Created at:           {fs.created_at}")
    if fs.created_by:
        console.print(f"  Created by:           {fs.created_by}")
    if fs.description:
        console.print(f"  Description:          {fs.description}")
    if fs.notes:
        console.print(f"  Notes:                {fs.notes}")


if __name__ == "__main__":
    main()
