#!/usr/bin/env python3
"""
Validate an experiment manifest against the pipeline state.

Checks that all referenced configs, data dirs, checkpoints, and signals
exist and are consistent with the manifest's declared parameters.

Usage:
    python scripts/validate_manifest.py experiments/nvda_hmhp_40feat_xnas_h10.yaml
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml required")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def validate(manifest_path: Path) -> list:
    """Validate manifest and return list of (level, message) tuples."""
    issues = []

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    if not manifest or "experiment" not in manifest:
        issues.append(("ERROR", "Missing 'experiment' section"))
        return issues

    exp = manifest["experiment"]
    if "name" not in exp:
        issues.append(("ERROR", "Missing experiment.name"))
    if "contract_version" not in exp:
        issues.append(("WARN", "Missing experiment.contract_version"))

    stages = manifest.get("stages", {})

    # Extraction
    extraction = stages.get("extraction", {})
    if extraction.get("enabled", False):
        config = extraction.get("config", "")
        if config:
            config_path = REPO_ROOT / config
            if not config_path.exists():
                issues.append(("ERROR", f"Extraction config not found: {config}"))
            else:
                issues.append(("OK", f"Extraction config exists: {config}"))

        output_dir = extraction.get("output_dir", "")
        if output_dir:
            full_path = REPO_ROOT / output_dir
            if full_path.exists():
                train_dir = full_path / "train"
                days = list(train_dir.glob("*_sequences.npy")) if train_dir.exists() else []
                if days:
                    issues.append(("OK", f"Extraction output: {len(days)} train days"))
                    first_meta = sorted(train_dir.glob("*_metadata.json"))
                    if first_meta:
                        try:
                            meta = json.load(open(first_meta[0]))
                            declared_fc = extraction.get("feature_count")
                            actual_fc = meta.get("n_features")
                            if declared_fc and actual_fc and declared_fc != actual_fc:
                                issues.append(("ERROR", f"Feature count mismatch: manifest={declared_fc}, data={actual_fc}"))
                            else:
                                issues.append(("OK", f"Feature count consistent: {actual_fc}"))

                            declared_ls = extraction.get("labeling_strategy")
                            actual_ls = meta.get("label_strategy")
                            if declared_ls and actual_ls and declared_ls != actual_ls:
                                issues.append(("ERROR", f"Label strategy mismatch: manifest={declared_ls}, data={actual_ls}"))
                        except Exception:
                            issues.append(("WARN", "Cannot read extraction metadata"))
                else:
                    issues.append(("WARN", f"Extraction output dir exists but no sequences: {output_dir}"))
            else:
                issues.append(("WARN", f"Extraction output not found: {output_dir}"))

    # Training
    training = stages.get("training", {})
    if training.get("enabled", False):
        config = training.get("config", "")
        if config:
            config_path = REPO_ROOT / config
            if not config_path.exists():
                issues.append(("ERROR", f"Training config not found: {config}"))
            else:
                issues.append(("OK", f"Training config exists: {config}"))

        output_dir = training.get("output_dir", "")
        if output_dir:
            full_path = REPO_ROOT / output_dir
            best_pt = full_path / "checkpoints" / "best.pt"
            if best_pt.exists():
                issues.append(("OK", f"Training checkpoint exists: best.pt"))
            else:
                issues.append(("WARN", f"No best.pt checkpoint in: {output_dir}"))

    # Signal export
    signal_export = stages.get("signal_export", {})
    if signal_export.get("enabled", False):
        output_dir = signal_export.get("output_dir", "")
        if "${" in output_dir:
            training_out = training.get("output_dir", "")
            output_dir = output_dir.replace("${stages.training.output_dir}", training_out)
        if output_dir:
            full_path = REPO_ROOT / output_dir
            if full_path.exists():
                pred = full_path / "predictions.npy"
                if pred.exists():
                    issues.append(("OK", f"Signals exported: predictions.npy exists"))
                else:
                    issues.append(("WARN", f"Signal dir exists but no predictions.npy"))
            else:
                issues.append(("WARN", f"Signals not exported: {output_dir}"))

    # Contract version
    contract_version = exp.get("contract_version", "")
    try:
        from hft_contracts import SCHEMA_VERSION
        if contract_version and contract_version != SCHEMA_VERSION:
            issues.append(("WARN", f"Contract version mismatch: manifest={contract_version}, current={SCHEMA_VERSION}"))
        else:
            issues.append(("OK", f"Contract version consistent: {SCHEMA_VERSION}"))
    except ImportError:
        issues.append(("WARN", "Cannot import hft_contracts to check schema version"))

    return issues


def main():
    parser = argparse.ArgumentParser(description="Validate Experiment Manifest")
    parser.add_argument("manifest", type=str, help="Path to manifest YAML")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        experiments_dir = Path(__file__).resolve().parent.parent / "experiments"
        manifest_path = experiments_dir / args.manifest
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {args.manifest}")
        sys.exit(1)

    print(f"Validating: {manifest_path.name}")
    print("=" * 50)

    issues = validate(manifest_path)

    errors = 0
    warnings = 0
    for level, msg in issues:
        prefix = {"OK": "  [OK]", "WARN": "  [!!]", "ERROR": "  [XX]"}.get(level, "  [??]")
        print(f"{prefix} {msg}")
        if level == "ERROR":
            errors += 1
        elif level == "WARN":
            warnings += 1

    print(f"\n  Result: {errors} errors, {warnings} warnings")
    sys.exit(1 if errors > 0 else 0)


if __name__ == "__main__":
    main()
