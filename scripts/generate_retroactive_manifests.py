#!/usr/bin/env python3
"""
Generate retroactive backfill manifests from existing experiment output directories.

Phase 1.3c of the training-pipeline-architecture migration: populate the
ledger with records for E1-E16-era experiments that completed before hft-ops
became the orchestrator. Without this, `hft-ops compare` would only surface
new experiments; most of the research history would be invisible.

This script:

  1. Scans ``{pipeline_root}/lob-model-trainer/outputs/experiments/`` for
     experiment output directories.
  2. Classifies each into a ``RecordType`` by inspecting artifacts
     (training_history.json → TRAINING; signals/ only → CALIBRATION; etc).
  3. Writes a lightweight metadata-only manifest under
     ``hft-ops/experiments/retroactive/<name>.yaml``.
  4. Optionally runs ``hft-ops ledger backfill`` for each generated manifest.

The generated manifests disable all stages (no training, no backtest) —
their sole purpose is to carry the experiment's name, tags, description,
and config reference into the ledger. Real artifacts remain on disk.

Usage:

    # Dry run (generate manifests only, don't backfill)
    python hft-ops/scripts/generate_retroactive_manifests.py

    # Generate AND backfill
    python hft-ops/scripts/generate_retroactive_manifests.py --backfill

    # Only process specific experiments
    python hft-ops/scripts/generate_retroactive_manifests.py \\
        --include e4_tlob_h60 e5_60s_huber_cvml --backfill

Idempotency: re-running overwrites the generated manifests but does not
re-backfill records that already exist in the ledger (dedup by fingerprint).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------------


def classify_experiment(output_dir: Path) -> Tuple[str, Optional[Path]]:
    """Return (record_type, metrics_file) for a directory on disk.

    record_type: one of ``training``, ``calibration``, ``sweep_aggregate``,
        ``analysis``, ``backtest``, ``evaluation``. Defaults to ``analysis``
        when nothing identifiable is present.
    metrics_file: absolute path to a JSON file the backfill CLI should consume,
        or None if no suitable file exists.
    """
    name = output_dir.name

    # Known sweep_aggregate patterns (single JSON with per-model results)
    for candidate in ("e4_baselines_H60.json", "e5_baselines_H10.json",
                      "ablation_results.json", "baselines.json"):
        if (output_dir / candidate).exists():
            return "sweep_aggregate", output_dir / candidate

    # Look for training_history.json (most common TRAINING case)
    history = output_dir / "training_history.json"
    if history.exists():
        return "training", history

    # Fallback: any results.json / metrics.json
    for candidate in ("results.json", "metrics.json", "training_metrics.json"):
        p = output_dir / candidate
        if p.exists():
            return "training", p

    # Calibration: E6-style, signals/ only, no training artifacts
    if (output_dir / "signals").exists() and not (output_dir / "checkpoints").exists():
        return "calibration", None

    # Checkpoint-only (training cancelled mid-flight): treat as training w/o metrics
    if (output_dir / "checkpoints").exists():
        return "training", None

    # Unclassifiable
    return "analysis", None


# -----------------------------------------------------------------------------
# Manifest generation
# -----------------------------------------------------------------------------


def generate_manifest(
    exp_name: str,
    record_type: str,
    trainer_configs_dir: Path,
    pipeline_root: Path,
) -> Dict:
    """Build a retroactive manifest dict for ``hft-ops/experiments/retroactive/``.

    The generated manifest has all stages disabled — it carries only metadata.
    A trainer config reference is included when the matching YAML exists in
    ``lob-model-trainer/configs/experiments/``, so the ledger can trace back
    to the code that produced the experiment.
    """
    trainer_cfg_path = trainer_configs_dir / f"{exp_name}.yaml"
    trainer_cfg_rel = ""
    if trainer_cfg_path.exists():
        trainer_cfg_rel = str(
            trainer_cfg_path.relative_to(pipeline_root)
        )

    # Try to read description / tags from the trainer config (if present)
    description = ""
    tags: List[str] = [f"retroactive-{record_type}"]
    if trainer_cfg_path.exists():
        try:
            import yaml
            with open(trainer_cfg_path, "r") as f:
                trainer_cfg = yaml.safe_load(f) or {}
            description = trainer_cfg.get("description", "").strip()
            trainer_tags = trainer_cfg.get("tags", [])
            if isinstance(trainer_tags, list):
                tags.extend(str(t) for t in trainer_tags if isinstance(t, (str, int)))
        except Exception:
            pass

    if not description:
        description = (
            f"Retroactive {record_type} record for '{exp_name}'. "
            f"Artifacts under lob-model-trainer/outputs/experiments/{exp_name}/."
        )

    manifest: Dict = {
        "experiment": {
            "name": exp_name,
            "description": description,
            "hypothesis": f"See EXPERIMENT_INDEX.md for the original hypothesis of {exp_name}.",
            "contract_version": "2.2",
            "tags": tags,
        },
        "pipeline_root": "..",
        "stages": {
            "extraction": {"enabled": False},
            "raw_analysis": {"enabled": False},
            "dataset_analysis": {"enabled": False},
            "training": {
                "enabled": False,  # historical; artifacts already on disk
                "config": trainer_cfg_rel if trainer_cfg_rel else "",
            },
            "signal_export": {"enabled": False},
            "backtesting": {"enabled": False},
        },
    }
    # If there's no trainer config, drop the config key (loader will complain otherwise
    # only if training.enabled=True; we set enabled=False so empty config is fine).
    if not trainer_cfg_rel:
        manifest["stages"]["training"].pop("config", None)

    return manifest


def write_manifest(manifest: Dict, path: Path) -> None:
    import yaml
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# =============================================================================\n"
        f"# Retroactive manifest for {manifest['experiment']['name']}\n"
        f"# Generated by hft-ops/scripts/generate_retroactive_manifests.py\n"
        f"# =============================================================================\n"
        f"# This manifest is METADATA-ONLY — all stages are disabled. Its purpose is to\n"
        f"# populate the ledger with a record pointing at pre-existing artifacts under\n"
        f"# lob-model-trainer/outputs/experiments/{manifest['experiment']['name']}/\n"
        f"# so `hft-ops compare` can surface historical results alongside new runs.\n"
        f"#\n"
        f"# Usage:\n"
        f"#   hft-ops ledger backfill experiments/retroactive/{manifest['experiment']['name']}.yaml \\\n"
        f"#       --metrics-file ../lob-model-trainer/outputs/experiments/{manifest['experiment']['name']}/training_history.json \\\n"
        f"#       --record-type <type>\n"
        f"# =============================================================================\n\n"
    )
    with open(path, "w") as f:
        f.write(header)
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------


def run_backfill(
    manifest_path: Path,
    record_type: str,
    metrics_file: Optional[Path],
    hft_ops_dir: Path,
) -> Tuple[bool, str]:
    """Invoke ``hft-ops ledger backfill`` for one manifest.

    Returns (success, output_message).
    """
    # Phase 6 6A.2 (2026-04-17): use current Python interpreter via `-m` invocation
    # instead of the previous hardcoded `.venv/bin/hft-ops` absolute path.
    # Original broke on Windows, non-venv installs, and any venv not named `.venv`.
    cmd = [
        sys.executable, "-m", "hft_ops.cli",
        "ledger", "backfill",
        str(manifest_path.relative_to(hft_ops_dir)),
        "--record-type", record_type,
        "--status", "completed",
        "--notes", "Generated by generate_retroactive_manifests.py",
    ]
    if metrics_file:
        cmd.extend(["--metrics-file", str(metrics_file)])

    proc = subprocess.run(
        cmd, cwd=str(hft_ops_dir), capture_output=True, text=True, timeout=60,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, output


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--pipeline-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent,
        help="HFT-pipeline-v2 root (default: auto-detected).",
    )
    ap.add_argument(
        "--backfill", action="store_true",
        help="After generating manifests, run `hft-ops ledger backfill` for each.",
    )
    ap.add_argument(
        "--include", nargs="+", default=None,
        help="Only process these experiment names (default: all).",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without writing files.",
    )
    args = ap.parse_args()

    pipeline_root = args.pipeline_root.resolve()
    outputs_dir = pipeline_root / "lob-model-trainer" / "outputs" / "experiments"
    trainer_configs_dir = pipeline_root / "lob-model-trainer" / "configs" / "experiments"
    hft_ops_dir = pipeline_root / "hft-ops"
    retroactive_dir = hft_ops_dir / "experiments" / "retroactive"

    if not outputs_dir.exists():
        print(f"ERROR: outputs_dir not found: {outputs_dir}", file=sys.stderr)
        sys.exit(1)

    candidates = sorted(
        d for d in outputs_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    )
    if args.include:
        include_set = set(args.include)
        candidates = [d for d in candidates if d.name in include_set]

    print(f"Scanning {len(candidates)} experiment output directories...")
    print()

    summary = {"generated": 0, "backfilled": 0, "skipped": 0, "failed": 0}
    results: List[Tuple[str, str, Optional[Path], bool, str]] = []

    for output_dir in candidates:
        exp_name = output_dir.name
        record_type, metrics_file = classify_experiment(output_dir)

        manifest = generate_manifest(
            exp_name, record_type, trainer_configs_dir, pipeline_root,
        )
        manifest_path = retroactive_dir / f"{exp_name}.yaml"

        if args.dry_run:
            print(f"  [dry] {exp_name} → {record_type} "
                  f"(metrics: {metrics_file.name if metrics_file else 'none'})")
            summary["generated"] += 1
            results.append((exp_name, record_type, metrics_file, True, "dry-run"))
            continue

        write_manifest(manifest, manifest_path)
        summary["generated"] += 1

        if args.backfill:
            ok, out = run_backfill(manifest_path, record_type, metrics_file, hft_ops_dir)
            results.append((exp_name, record_type, metrics_file, ok, out))
            if ok:
                summary["backfilled"] += 1
            elif "already exists" in out:
                summary["skipped"] += 1
            else:
                summary["failed"] += 1
        else:
            results.append((exp_name, record_type, metrics_file, True, "(no backfill)"))

    # Report
    print()
    print(f"Generated: {summary['generated']}")
    if args.backfill:
        print(f"Backfilled: {summary['backfilled']}")
        print(f"Skipped (dedup): {summary['skipped']}")
        print(f"Failed: {summary['failed']}")

    if summary["failed"]:
        print()
        print("Failures:")
        for name, rtype, mfile, ok, out in results:
            if not ok and "already exists" not in out:
                print(f"  {name} ({rtype}):")
                for line in out.splitlines()[-5:]:
                    print(f"    {line}")


if __name__ == "__main__":
    main()
