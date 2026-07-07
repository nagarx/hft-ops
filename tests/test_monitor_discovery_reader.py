"""TDD — the DiscoveryVerdictReader: rglob */results/*.json across the discovery
trees, skip internals/denylist, normalize each via the discovery_verdict adapters.
"""

import shutil
from pathlib import Path

from discovery_verdict import Verdict

from hft_ops.monitor.discovery_reader import DiscoveryVerdictReader

FIX = Path(__file__).parent / "fixtures" / "monitor_discovery"


def _put(tmp_path, rel, name):
    d = tmp_path / rel
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIX / name, d / name)


def test_skips_internals_and_denylist_and_tags_tree(tmp_path):
    _put(tmp_path, "glbx_discovery/results", "tradeflow_esnq_P_A7.json")
    _put(tmp_path, "glbx_discovery/results", "_subsec_validate.json")     # underscore-internal -> skip
    _put(tmp_path, "glbx_discovery/results", "gex_features.json")          # denylisted dump -> skip
    _put(tmp_path, "opra_discovery/gex_variance/results", "gex_variance_verdict.json")  # nested

    verdicts = DiscoveryVerdictReader(tmp_path).read_all()
    names = {Path(v.source_path).name for v in verdicts}
    assert names == {"tradeflow_esnq_P_A7.json", "gex_variance_verdict.json"}
    assert all(isinstance(v, Verdict) for v in verdicts)
    assert {v.source_tree for v in verdicts} == {"glbx_discovery", "opra_discovery"}

    tf = next(v for v in verdicts if v.source_tree == "glbx_discovery")
    assert tf.verdict == "FAIL"
    assert tf.adapter_name == "CommonCoreAdapter"


def test_include_gate_outs_false_drops_gated_out(tmp_path):
    _put(tmp_path, "glbx_discovery/results", "tradeflow_esnq_P_A7.json")
    _put(tmp_path, "glbx_discovery/results", "gate0_diagnosis.json")

    with_gate = DiscoveryVerdictReader(tmp_path, include_gate_outs=True).read_all()
    without = DiscoveryVerdictReader(tmp_path, include_gate_outs=False).read_all()

    assert any(v.verdict == "GATED_OUT" for v in with_gate)
    assert all(v.verdict != "GATED_OUT" for v in without)
    assert {Path(v.source_path).name for v in without} == {"tradeflow_esnq_P_A7.json"}


def test_skips_sharded_data_caches(tmp_path):
    # The iv_shadow ATM-IV cache (``*.shard*of*.json``, shape ``{"days": ...}``)
    # is co-located in a harness ``results/`` dir but is NOT a verdict. Without the
    # SKIP_GLOBS guard it falls through to the CommonCoreAdapter catch-all and
    # injects a phantom UNRESOLVED row into the monitor.
    _put(tmp_path, "nvda_discovery/iv_shadow/results", "tradeflow_esnq_P_A7.json")  # real verdict
    _put(tmp_path, "nvda_discovery/iv_shadow/results", "foo.shard0of4.json")        # data cache -> skip

    verdicts = DiscoveryVerdictReader(tmp_path).read_all()
    got = {Path(v.source_path).name for v in verdicts}
    expected = {"tradeflow_esnq_P_A7.json"}
    assert got == expected, (
        f"shard cache must not become a verdict. expected {expected}, got {got}"
    )


def test_missing_trees_tolerated(tmp_path):
    reader = DiscoveryVerdictReader(tmp_path)   # no discovery trees present at all
    assert reader.read_all() == []
    assert reader.read_errors == []


def test_void_and_non_verdict_artifacts_denylisted(tmp_path):
    # Phase-2 TRUTH (2026-07-07): the VOID superseded verdict
    # (variance_dl_verdict.json — FINDING-110 / results/SUPERSEDED.md) and the 5
    # known non-verdict result artifacts must be name-skipped by DEFAULT_DENYLIST,
    # never surfaced as monitor rows; the CORRECTED variance_dl_v2_verdict.json
    # must survive.
    res = "nvda_discovery/variance_dl/results"
    _put(tmp_path, res, "tradeflow_esnq_P_A7.json")  # unrelated real verdict — survives
    d = tmp_path / res
    # Corrected v2 verdict (real verdict shape) — must NOT be denylisted.
    shutil.copy(FIX / "gex_variance_verdict.json", d / "variance_dl_v2_verdict.json")
    denylisted = [
        "variance_dl_verdict.json",              # VOID STOP verdict (class 2)
        "composite_vrp_confront_env_gates.json", # class 3 — non-verdict artifacts
        "frozen_scale_model.json",
        "strike_grids.json",
        "nvda_0dte_iv.json",
        "execution_timing_curve.json",
    ]
    for name in denylisted:
        (d / name).write_text('{"note": "must never become a monitor row"}')

    reader = DiscoveryVerdictReader(tmp_path)
    verdicts = reader.read_all()
    got = {Path(v.source_path).name for v in verdicts}
    expected = {"tradeflow_esnq_P_A7.json", "variance_dl_v2_verdict.json"}
    assert got == expected, (
        f"denylist must drop the VOID verdict + non-verdict artifacts. "
        f"expected {expected}, got {got}"
    )
    assert reader.read_errors == []  # denylist skip happens BEFORE parsing


def test_default_denylist_covers_known_hazards():
    # Regression lock on the DEFAULT_DENYLIST contents: the VOID variance-DL STOP
    # verdict + the non-verdict JSONs that live in scanned results/ dirs today.
    must_deny = {
        "variance_dl_verdict.json",
        "composite_vrp_confront_env_gates.json",
        "frozen_scale_model.json",
        "strike_grids.json",
        "nvda_0dte_iv.json",
        "nvda_0dte_atm_iv.json",
        "execution_timing_curve.json",
    }
    assert must_deny <= set(DiscoveryVerdictReader.DEFAULT_DENYLIST)
    # The corrected verdict must never be denylisted.
    assert "variance_dl_v2_verdict.json" not in DiscoveryVerdictReader.DEFAULT_DENYLIST


def test_path_guard_skips_venv_and_site_packages(tmp_path):
    # crypto_discovery/.venv ships statsmodels test fixtures matching
    # **/results/*.json (verified on disk 2026-07-07). The SKIP_PATH_PARTS
    # guard must drop any hit under a .venv/ or site-packages/ path component
    # BEFORE parsing — even INVALID JSON there must neither become a row nor
    # a read_error.
    _put(tmp_path, "crypto_discovery/variance_probe/results", "tradeflow_esnq_P_A7.json")
    venv = tmp_path / "crypto_discovery/.venv/lib/python3.14/site-packages/statsmodels/tsa/tests/results"
    venv.mkdir(parents=True)
    (venv / "fit_ets_results.json").write_text('{"TRUE": 1, "FALSE": 2}')  # parseable non-verdict
    (venv / "broken_fixture.json").write_text("{not json")                  # must not even error
    sp = tmp_path / "multiday_discovery/env/lib/site-packages/pkg/tests/results"
    sp.mkdir(parents=True)  # site-packages WITHOUT .venv — proves the 2nd component independently
    (sp / "some_fixture.json").write_text('{"days": []}')

    reader = DiscoveryVerdictReader(tmp_path)
    verdicts = reader.read_all()
    got = {Path(v.source_path).name for v in verdicts}
    assert got == {"tradeflow_esnq_P_A7.json"}, (
        f"venv/site-packages hits must never become rows. got {got}"
    )
    assert reader.read_errors == [], (
        "guard must skip BEFORE parsing — a broken venv fixture must not "
        f"surface in read_errors. got {reader.read_errors}"
    )


def test_crypto_and_multiday_trees_scanned(tmp_path):
    # Phase-3 (2026-07-07): crypto_discovery + multiday_discovery are IN scope.
    _put(tmp_path, "crypto_discovery/reversion_probe/results", "tradeflow_esnq_P_A7.json")
    _put(tmp_path, "multiday_discovery/vrp_0dte/results", "gex_variance_verdict.json")
    verdicts = DiscoveryVerdictReader(tmp_path).read_all()
    assert {v.source_tree for v in verdicts} == {"crypto_discovery", "multiday_discovery"}


def test_discovery_trees_membership_lock():
    # Regression lock on the scan scope (count + membership + order stability).
    # Adding/removing a tree is a deliberate scope decision — update this lock
    # in the same commit as the docstring + CODEBASE §2.10 + README statements.
    assert DiscoveryVerdictReader.DISCOVERY_TREES == (
        "glbx_discovery",
        "xsec_equity_discovery",
        "nvda_discovery",
        "opra_discovery",
        "pead_discovery",
        "crypto_discovery",
        "multiday_discovery",
    )
    assert len(DiscoveryVerdictReader.DISCOVERY_TREES) == 7
    assert DiscoveryVerdictReader.SKIP_PATH_PARTS == (".venv", "site-packages")


def test_multiday_iv_panel_cache_denylisted(tmp_path):
    # multiday_discovery/vrp_0dte/results/nvda_0dte_atm_iv.json is an IV data
    # panel (shape {"days": ...}), NOT a verdict — the class-3 sibling of the
    # already-denylisted nvda_0dte_iv.json. Without its DEFAULT_DENYLIST entry
    # it falls through to the CommonCoreAdapter catch-all as a phantom
    # UNRESOLVED row (verified against the live file 2026-07-07).
    _put(tmp_path, "multiday_discovery/vrp_0dte/results", "gex_variance_verdict.json")
    d = tmp_path / "multiday_discovery/vrp_0dte/results"
    (d / "nvda_0dte_atm_iv.json").write_text('{"days": {}, "config": {}, "n_days_ok": 0}')
    verdicts = DiscoveryVerdictReader(tmp_path).read_all()
    assert {Path(v.source_path).name for v in verdicts} == {"gex_variance_verdict.json"}
