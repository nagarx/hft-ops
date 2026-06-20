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


def test_missing_trees_tolerated(tmp_path):
    reader = DiscoveryVerdictReader(tmp_path)   # no discovery trees present at all
    assert reader.read_all() == []
    assert reader.read_errors == []
