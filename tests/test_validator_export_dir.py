"""Coverage for ``_validate_existing_exports`` delegating to the
``hft_contracts.validate_export_dir`` SSoT (Foundation Integrity, 2026-05-29).

The prior implementation sampled only the first day-metadata of the first subdir
then ``break``-ed, so it could not see manifest<->disk count drift or cross-day
schema/commit pollution — and it had no test coverage with a populated export
dir. This locks the consolidated delegation + the populated/empty guard.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

from hft_ops.manifest.validator import ValidationResult, _validate_existing_exports

GOOD = "c62a1c0d9ed1b9b75dc9dabace78bf51d78ceead"
ALT = "b5e746dfc4f0023d6650ca8ad12b52db797fe1fa"


def _meta(day, *, schema="3.0", commit=GOOD, n=100):
    return {
        "day": day,
        "n_sequences": n,
        "n_features": 98,
        "window_size": 20,
        "schema_version": schema,
        "contract_version": schema,
        "label_strategy": "tlob",
        "label_encoding": {"down": -1, "stable": 0, "up": 1},
        "normalization": {"strategy": "none"},
        "provenance": {"extractor_version": "0.1.0", "git_commit": commit, "git_dirty": False},
        "horizons": [10, 60, 300],
        "export_timestamp": "2026-05-29T00:00:00Z",
    }


def _build(root, splits, *, schema="3.0", commit=GOOD, schema_overrides=None,
           commit_overrides=None, manifest=None, write_manifest=True):
    schema_overrides = schema_overrides or {}
    commit_overrides = commit_overrides or {}
    for split, days in splits.items():
        sd = root / split
        sd.mkdir(parents=True, exist_ok=True)
        for day in days:
            (sd / f"{day}_sequences.npy").write_bytes(b"")
            (sd / f"{day}_metadata.json").write_text(json.dumps(_meta(
                day, schema=schema_overrides.get(day, schema),
                commit=commit_overrides.get(day, commit))))
    if write_manifest:
        dm = {
            "schema_version": schema,
            "days_processed": sum(len(d) for d in splits.values()),
            "total_sequences": 999_999,  # pre-align — must be ignored (CF-1)
            "split": {s: {"days": len(d), "sequences": 1} for s, d in splits.items()},
            "partial_failure": {"status": "complete", "failed_partitions": []},
            "skipped_days": 0,
        }
        if manifest:
            dm.update(manifest)
        (root / "dataset_manifest.json").write_text(json.dumps(dm))
    return root


class _StubPaths:
    def __init__(self, target: Path):
        self._t = target

    def resolve(self, _rel):
        return self._t


def _manifest(output_dir="export"):
    return types.SimpleNamespace(
        stages=types.SimpleNamespace(
            extraction=types.SimpleNamespace(output_dir=output_dir)
        )
    )


def _run(target: Path, output_dir="export") -> ValidationResult:
    result = ValidationResult()
    _validate_existing_exports(_manifest(output_dir), _StubPaths(target), result)
    return result


class TestValidateExistingExportsDelegation:
    def test_clean_export_no_errors(self, tmp_path):
        _build(tmp_path, {"train": ["d1", "d2"], "val": ["d3"], "test": ["d4"]})
        result = _run(tmp_path)
        assert result.is_valid, [str(e) for e in result.errors]

    def test_polluted_export_surfaces_errors(self, tmp_path):
        _build(
            tmp_path,
            {"train": ["d1", "d2", "d3"], "val": ["d4"], "test": ["d5"]},
            schema_overrides={"d3": "2.2"},
            commit_overrides={"d3": ALT},
            manifest={
                "days_processed": 5,  # total disk = 5 → total check passes
                "split": {  # claims train=2 but disk has 3 (stale leftover)
                    "train": {"days": 2, "sequences": 1},
                    "val": {"days": 1, "sequences": 1},
                    "test": {"days": 1, "sequences": 1},
                },
            },
        )
        result = _run(tmp_path)
        assert not result.is_valid
        joined = "\n".join(str(e) for e in result.errors)
        assert "schema_version" in joined  # mixed {2.2, 3.0}
        assert "git_commit" in joined      # mixed producer commit
        assert "count" in joined           # extra stale file

    def test_empty_output_dir_string_skips(self, tmp_path):
        result = _run(tmp_path, output_dir="")
        assert result.is_valid
        assert not result.errors

    def test_nonexistent_output_dir_skips(self, tmp_path):
        result = _run(tmp_path / "nope")
        assert result.is_valid
        assert not result.errors

    def test_truly_empty_dir_skips(self, tmp_path):
        # exists, but no manifest and no day-data → nothing extracted yet
        result = _run(tmp_path)
        assert result.is_valid
        assert not result.errors

    def test_daydata_without_manifest_errors(self, tmp_path):
        # day-data present but manifest missing → defective MBO export, must flag
        _build(tmp_path, {"train": ["d1"], "val": ["d2"], "test": ["d3"]}, write_manifest=False)
        result = _run(tmp_path)
        assert not result.is_valid
        assert any("dataset_manifest" in str(e) for e in result.errors)
