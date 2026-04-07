#!/usr/bin/env python3
"""
Unified Experiment Dashboard.

Reads all experiment manifests in hft-ops/experiments/ and checks the
pipeline state for each: extraction done? training done? signals exported?
backtest run? Produces a unified view across the entire pipeline.

Usage:
    python scripts/experiment_dashboard.py
    python scripts/experiment_dashboard.py --manifest experiments/nvda_hmhp_40feat_xnas_h10.yaml
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml required. pip install pyyaml")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def check_extraction(manifest: dict) -> dict:
    """Check if extraction stage output exists."""
    stages = manifest.get("stages", {})
    extraction = stages.get("extraction", {})
    output_dir = extraction.get("output_dir", "")
    if not output_dir:
        return {"status": "not_configured", "details": ""}

    full_path = REPO_ROOT / output_dir
    if not full_path.exists():
        return {"status": "missing", "details": f"{output_dir} not found"}

    train_dir = full_path / "train"
    days = list(train_dir.glob("*_sequences.npy")) if train_dir.exists() else []
    manifest_path = full_path / "dataset_manifest.json"
    has_manifest = manifest_path.exists()

    if len(days) == 0:
        return {"status": "empty", "details": "No sequence files"}

    details = f"{len(days)} train days"
    if has_manifest:
        try:
            m = json.load(open(manifest_path))
            details += f", {m.get('total_sequences', '?')} sequences"
        except Exception:
            pass

    return {"status": "done", "details": details}


def check_training(manifest: dict) -> dict:
    """Check if training stage output exists."""
    stages = manifest.get("stages", {})
    training = stages.get("training", {})
    output_dir = training.get("output_dir", "")
    if not output_dir:
        return {"status": "not_configured", "details": ""}

    full_path = REPO_ROOT / output_dir
    if not full_path.exists():
        return {"status": "missing", "details": f"{output_dir} not found"}

    best_pt = full_path / "checkpoints" / "best.pt"
    history = full_path / "training_history.json"
    config = full_path / "config.yaml"

    if not best_pt.exists():
        return {"status": "incomplete", "details": "No best.pt checkpoint"}

    details = "checkpoint exists"

    log_dir = full_path / "logs"
    if log_dir.exists():
        log_files = list(log_dir.glob("*.log"))
        if log_files:
            last_log = sorted(log_files)[-1]
            try:
                with open(last_log) as f:
                    lines = f.readlines()
                for line in reversed(lines):
                    if "val_h10_accuracy" in line:
                        import re
                        m = re.search(r"val_h10_accuracy=([\d.]+)", line)
                        if m:
                            details += f", H10 val={float(m.group(1))*100:.2f}%"
                        break
            except Exception:
                pass

    confidence_file = full_path / "confidence_analysis.json"
    if confidence_file.exists():
        try:
            ca = json.load(open(confidence_file))
            for split_data in ca.get("splits", {}).values():
                for m in split_data.get("conditioned_metrics", []):
                    if m.get("filter") == "agreement=1.0":
                        details += f", agree=1.0 dir_acc={m.get('dir_acc', 0):.2%}"
                        break
        except Exception:
            pass

    return {"status": "done", "details": details}


def check_signals(manifest: dict) -> dict:
    """Check if signal export exists."""
    stages = manifest.get("stages", {})
    signal_export = stages.get("signal_export", {})
    output_dir = signal_export.get("output_dir", "")
    if not output_dir:
        return {"status": "not_configured", "details": ""}

    output_dir = output_dir.replace("${stages.training.output_dir}",
                                     stages.get("training", {}).get("output_dir", ""))
    full_path = REPO_ROOT / output_dir
    if not full_path.exists():
        return {"status": "missing", "details": ""}

    predictions = full_path / "predictions.npy"
    metadata = full_path / "signal_metadata.json"

    if not predictions.exists():
        return {"status": "incomplete", "details": "No predictions.npy"}

    details = ""
    if metadata.exists():
        try:
            sm = json.load(open(metadata))
            details = f"{sm.get('total_samples', '?')} samples"
            details += f", {sm.get('agreement_distribution', {}).get('full_agreement', '?')} agree=1.0"
        except Exception:
            pass

    return {"status": "done", "details": details}


def check_backtests(manifest: dict) -> dict:
    """Check if backtest results exist."""
    backtest_dir = REPO_ROOT / "lob-backtester" / "outputs" / "backtests"
    if not backtest_dir.exists():
        return {"status": "not_run", "details": ""}

    exp_name = manifest.get("experiment", {}).get("name", "")
    index_path = backtest_dir / "index.json"
    if not index_path.exists():
        return {"status": "not_run", "details": "No backtest registry"}

    try:
        index = json.load(open(index_path))
    except Exception:
        return {"status": "error", "details": "Cannot read index.json"}

    matching = []
    for run_id, meta in index.items():
        if exp_name and exp_name.replace("nvda_hmhp_", "") in run_id:
            matching.append((run_id, meta))

    if not matching:
        return {"status": "not_run", "details": "No matching backtest runs"}

    details_parts = []
    for run_id, meta in matching[:3]:
        name = meta.get("name", run_id)[:25]
        ret = meta.get("total_return", 0)
        details_parts.append(f"{name}: {ret:.2%}")

    return {"status": "done", "details": "; ".join(details_parts)}


def display_experiment(manifest_path: Path, manifest: dict) -> None:
    """Display status for one experiment."""
    exp = manifest.get("experiment", {})
    name = exp.get("name", manifest_path.stem)
    desc = exp.get("description", "").strip()[:80]
    tags = exp.get("tags", [])

    extraction = check_extraction(manifest)
    training = check_training(manifest)
    signals = check_signals(manifest)
    backtests = check_backtests(manifest)

    status_icon = {
        "done": "[OK]", "missing": "[--]", "incomplete": "[..]",
        "not_configured": "[NC]", "not_run": "[--]", "empty": "[--]",
        "error": "[!!]",
    }

    print(f"\n  {name}")
    print(f"  {'='*len(name)}")
    if desc:
        print(f"  {desc}")
    if tags:
        print(f"  Tags: {', '.join(tags[:6])}")
    print()
    print(f"  Extraction:  {status_icon.get(extraction['status'], '[??]')} {extraction['details']}")
    print(f"  Training:    {status_icon.get(training['status'], '[??]')} {training['details']}")
    print(f"  Signals:     {status_icon.get(signals['status'], '[??]')} {signals['details']}")
    print(f"  Backtesting: {status_icon.get(backtests['status'], '[??]')} {backtests['details']}")

    # Profiler reference
    profiler = manifest.get("profiler_references", {})
    stats = profiler.get("statistical_basis", {})
    if stats:
        parts = []
        if "ofi_signal_r_h10" in stats:
            parts.append(f"OFI r={stats['ofi_signal_r_h10']}")
        if "vwes_bps" in stats:
            parts.append(f"VWES={stats['vwes_bps']} bps")
        if parts:
            print(f"  Profiler:    {', '.join(parts)}")


def main():
    parser = argparse.ArgumentParser(description="Experiment Dashboard")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Show status for a specific manifest only")
    args = parser.parse_args()

    print("=" * 60)
    print("  EXPERIMENT DASHBOARD")
    print("=" * 60)

    experiments_dir = Path(__file__).resolve().parent.parent / "experiments"

    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            manifest_path = experiments_dir / args.manifest
        if not manifest_path.exists():
            print(f"ERROR: Manifest not found: {args.manifest}")
            sys.exit(1)
        manifests = [(manifest_path, yaml.safe_load(open(manifest_path)))]
    else:
        manifests = []
        for f in sorted(experiments_dir.glob("*.yaml")):
            try:
                data = yaml.safe_load(open(f))
                if data and "experiment" in data:
                    manifests.append((f, data))
            except Exception:
                pass

    if not manifests:
        print("\n  No experiment manifests found.")
        sys.exit(0)

    print(f"\n  Found {len(manifests)} experiment manifest(s)")

    for manifest_path, manifest in manifests:
        display_experiment(manifest_path, manifest)

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
