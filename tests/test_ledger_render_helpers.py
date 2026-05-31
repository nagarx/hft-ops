"""Steps 4 + 5-render (2026-05-31): pure formatters that surface the
Foundation-Integrity producer_commits + the validation/post-training gate_reports
in `ledger show`, and the per-key producer_commits divergence in `diff`.

These were DARK: `ledger show` rendered only git short-hash + training/backtest
metrics (producer_commits + gate_reports never shown), and `diff` never surfaced
producer_commits. Pure helpers (lines / rows) so the render logic is unit-tested
without a CliRunner+ledger fixture (matches the `--stages` "test the pure helper"
convention).
"""

from __future__ import annotations

from hft_ops.cli import (
    _format_dataset_health,
    _format_gate_reports,
    _format_producer_commits,
    _format_producer_commits_divergence,
    _short_provenance_val,
)


# --------------------------------------------------------------------------
# _short_provenance_val
# --------------------------------------------------------------------------
def test_short_val_none_is_placeholder():
    assert _short_provenance_val(None) == "(none)"


def test_short_val_hex_is_truncated():
    assert _short_provenance_val("a" * 40) == "a" * 12 + "..."


def test_short_val_unresolved_passthrough():
    assert _short_provenance_val("unresolved") == "unresolved"


def test_short_val_short_string_passthrough():
    assert _short_provenance_val("git-pin") == "git-pin"


# --------------------------------------------------------------------------
# _format_producer_commits (ledger show)
# --------------------------------------------------------------------------
def test_producer_commits_empty_no_lines():
    assert _format_producer_commits({}) == []


def test_producer_commits_renders_completeness_shas_and_source():
    pc = {
        "extractor_git_sha": "a" * 40,
        "reconstructor_git_sha": "b" * 40,
        "hft_statistics_git_sha": "c" * 40,
        "reconstructor_source": "path-override@" + "b" * 40 + "+clean",
        "completeness": "full",
    }
    joined = "\n".join(_format_producer_commits(pc))
    assert "completeness=full" in joined
    assert "extractor_git_sha: " + "a" * 12 + "..." in joined
    assert "reconstructor_git_sha: " + "b" * 12 + "..." in joined
    assert "hft_statistics_git_sha: " + "c" * 12 + "..." in joined
    assert "reconstructor_source: path-override@" in joined


def test_producer_commits_unresolved_passthrough():
    pc = {"reconstructor_git_sha": "unresolved", "completeness": "partial"}
    joined = "\n".join(_format_producer_commits(pc))
    assert "completeness=partial" in joined
    assert "reconstructor_git_sha: unresolved" in joined  # NOT truncated


# --------------------------------------------------------------------------
# _format_gate_reports (ledger show)
# --------------------------------------------------------------------------
def test_gate_reports_empty_no_lines():
    assert _format_gate_reports({}) == []


def test_gate_reports_renders_status_and_ic_scalars():
    gr = {
        "validation": {
            "status": "pass",
            "best_feature_ic": 0.248,
            "ic_count": 3,
            "return_std_bps": 12.5,
            "stability": 15.2,
            "summary": "IC gate passed",
        }
    }
    joined = "\n".join(_format_gate_reports(gr))
    assert "Gate Reports" in joined
    assert "validation: pass" in joined
    assert "best_feature_ic: 0.2480" in joined  # float .4f
    assert "ic_count: 3" in joined
    assert "return_std_bps: 12.5000" in joined
    assert "summary: IC gate passed" in joined


def test_gate_reports_bool_field():
    gr = {"post_training_gate": {"status": "warn", "regressed": True}}
    joined = "\n".join(_format_gate_reports(gr))
    assert "post_training_gate: warn" in joined
    assert "regressed: True" in joined


def test_gate_reports_only_non_dict_reports_returns_empty():
    # malformed (no valid dict report) -> no header, no body
    assert _format_gate_reports({"x": "not a dict"}) == []


def test_gate_reports_long_summary_truncated():
    gr = {"validation": {"status": "pass", "summary": "x" * 300}}
    joined = "\n".join(_format_gate_reports(gr))
    assert "…" in joined  # truncated
    assert "x" * 300 not in joined


# --------------------------------------------------------------------------
# _format_producer_commits_divergence (diff)
# --------------------------------------------------------------------------
def test_divergence_differing_reconstructor_sha():
    pc_a = {"extractor_git_sha": "a" * 40, "reconstructor_git_sha": "b" * 40}
    pc_b = {"extractor_git_sha": "a" * 40, "reconstructor_git_sha": "c" * 40}
    rows = _format_producer_commits_divergence(pc_a, pc_b)
    assert len(rows) == 1
    key, va, vb = rows[0]
    assert key == "reconstructor_git_sha"
    assert va == "b" * 12 + "..." and vb == "c" * 12 + "..."


def test_divergence_identical_no_rows():
    pc = {"extractor_git_sha": "a" * 40}
    assert _format_producer_commits_divergence(pc, dict(pc)) == []


def test_divergence_asymmetric_key_uses_none_placeholder():
    rows = _format_producer_commits_divergence({"extractor_git_sha": "a" * 40}, {})
    assert len(rows) == 1
    key, va, vb = rows[0]
    assert key == "extractor_git_sha"
    assert vb == "(none)"


# --------------------------------------------------------------------------
# _format_dataset_health (ledger show) — Step 6
# --------------------------------------------------------------------------
def test_dataset_health_empty_no_lines():
    assert _format_dataset_health({}) == []


def test_dataset_health_renders_summary():
    joined = "\n".join(_format_dataset_health({
        "report_dir": "/x/r",
        "analyzers": ["data_quality", "return_analysis"],
        "split": "train",
    }))
    assert "Dataset Health" in joined
    assert "report_dir: /x/r" in joined
    assert "analyzers: data_quality, return_analysis" in joined  # list joined
    assert "split: train" in joined
