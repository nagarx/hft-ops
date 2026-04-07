#!/usr/bin/env python3
"""
Cross-experiment comparison table.

Reads experiment manifests and their results to produce a side-by-side
comparison of model accuracy, readability metrics, and backtest performance.

Usage:
    python scripts/compare_experiments.py
    python scripts/compare_experiments.py --format markdown
"""

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml required")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def extract_training_metrics(manifest: dict) -> dict:
    """Extract training metrics from log files."""
    training = manifest.get("stages", {}).get("training", {})
    output_dir = training.get("output_dir", "")
    if not output_dir:
        return {}

    full_path = REPO_ROOT / output_dir
    result = {}

    log_dir = full_path / "logs"
    if log_dir.exists():
        log_files = sorted(log_dir.glob("*.log"))
        if log_files:
            try:
                with open(log_files[-1]) as f:
                    content = f.read()
                h10_matches = re.findall(r"val_h10_accuracy=([\d.]+)", content)
                if h10_matches:
                    result["h10_val"] = float(h10_matches[-1])
                loss_matches = re.findall(r"val_loss=([\d.]+)", content)
                if loss_matches:
                    result["val_loss"] = float(loss_matches[-1])
                epoch_matches = re.findall(r"Epoch (\d+):", content)
                if epoch_matches:
                    result["epochs"] = int(epoch_matches[-1])
            except Exception:
                pass

    confidence_file = full_path / "confidence_analysis.json"
    if confidence_file.exists():
        try:
            ca = json.load(open(confidence_file))
            for split_name, split_data in ca.get("splits", {}).items():
                if split_name == "test":
                    for m in split_data.get("conditioned_metrics", []):
                        if m.get("filter") == "all_samples":
                            result["test_acc"] = m.get("accuracy", 0)
                            result["test_dir_acc"] = m.get("dir_acc", 0)
                        elif "agree=1.0 AND confirm>0.65" in m.get("filter", ""):
                            result["high_conv_acc"] = m.get("accuracy", 0)
                            result["high_conv_dir_acc"] = m.get("dir_acc", 0)
                            result["high_conv_rate"] = m.get("rate", 0)
        except Exception:
            pass

    spread_file = full_path / "spread_analysis.json"
    if spread_file.exists():
        try:
            sa = json.load(open(spread_file))
            for split_name, split_data in sa.get("splits", {}).items():
                if split_name == "test":
                    for m in split_data.get("conditioned_metrics", []):
                        if m.get("filter") == "FULL_READABILITY: high_conf+1tick+dir":
                            result["readability_dir_acc"] = m.get("dir_acc", 0)
                            result["readability_rate"] = m.get("rate", 0)
        except Exception:
            pass

    return result


def extract_backtest_metrics(manifest: dict) -> dict:
    """Extract backtest metrics from registry."""
    backtest_dir = REPO_ROOT / "lob-backtester" / "outputs" / "backtests"
    index_path = backtest_dir / "index.json"
    if not index_path.exists():
        return {}

    try:
        index = json.load(open(index_path))
    except Exception:
        return {}

    exp_name = manifest.get("experiment", {}).get("name", "")
    results = {}

    for run_id, meta in index.items():
        name = meta.get("name", "")
        ret = meta.get("total_return", 0)
        trades = meta.get("total_trades", 0)
        if "h10" in name and ("40feat" in name or "40feat" in exp_name):
            results["bt_h10_return"] = ret
            results["bt_h10_trades"] = trades
        elif "h60" in name:
            results["bt_h60_return"] = ret
        elif "h300" in name:
            results["bt_h300_return"] = ret

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare Experiments")
    parser.add_argument("--format", type=str, default="table", choices=["table", "markdown"])
    args = parser.parse_args()

    experiments_dir = Path(__file__).resolve().parent.parent / "experiments"
    manifests = []
    for f in sorted(experiments_dir.glob("nvda_hmhp_*.yaml")):
        try:
            data = yaml.safe_load(open(f))
            if data and "experiment" in data:
                manifests.append((f, data))
        except Exception:
            pass

    if not manifests:
        print("No HMHP experiment manifests found.")
        sys.exit(0)

    rows = []
    for manifest_path, manifest in manifests:
        exp = manifest.get("experiment", {})
        training = manifest.get("stages", {}).get("training", {})
        profiler = manifest.get("profiler_references", {}).get("statistical_basis", {})

        t_metrics = extract_training_metrics(manifest)
        b_metrics = extract_backtest_metrics(manifest)

        rows.append({
            "name": exp.get("name", "")[:30],
            "features": training.get("input_size", "?"),
            "preset": training.get("feature_preset", "?")[:15],
            "exchange": exp.get("tags", ["?"])[1] if len(exp.get("tags", [])) > 1 else "?",
            "h10_val": f"{t_metrics.get('h10_val', 0)*100:.2f}%" if t_metrics.get("h10_val") else "N/A",
            "high_conv_dir": f"{t_metrics.get('high_conv_dir_acc', 0)*100:.1f}%" if t_metrics.get("high_conv_dir_acc") else "N/A",
            "readability_dir": f"{t_metrics.get('readability_dir_acc', 0)*100:.1f}%" if t_metrics.get("readability_dir_acc") else "N/A",
            "vwes": f"{profiler.get('vwes_bps', '?')} bps",
            "bt_h10": f"{b_metrics.get('bt_h10_return', 0)*100:.2f}%" if b_metrics.get("bt_h10_return") is not None else "N/A",
            "bt_h60": f"{b_metrics.get('bt_h60_return', 0)*100:.2f}%" if b_metrics.get("bt_h60_return") is not None else "N/A",
        })

    if args.format == "markdown":
        print("| Experiment | Features | Preset | Exchange | H10 Val | High Conv Dir | Readability Dir | VWES | BT H10 | BT H60 |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['name']} | {r['features']} | {r['preset']} | {r['exchange']} | "
                  f"{r['h10_val']} | {r['high_conv_dir']} | {r['readability_dir']} | "
                  f"{r['vwes']} | {r['bt_h10']} | {r['bt_h60']} |")
    else:
        header = f"{'Experiment':<32} {'Feat':>4} {'Exchange':>8} {'H10 Val':>8} {'HiConv Dir':>10} {'Read Dir':>9} {'VWES':>7} {'BT H10':>8} {'BT H60':>8}"
        print(header)
        print("-" * len(header))
        for r in rows:
            print(f"{r['name']:<32} {r['features']:>4} {r['exchange']:>8} {r['h10_val']:>8} "
                  f"{r['high_conv_dir']:>10} {r['readability_dir']:>9} {r['vwes']:>7} "
                  f"{r['bt_h10']:>8} {r['bt_h60']:>8}")


if __name__ == "__main__":
    main()
