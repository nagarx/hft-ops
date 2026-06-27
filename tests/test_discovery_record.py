"""TDD — record_from_verdict: map a discovery ``Verdict`` to a fingerprinted,
first-class ``ExperimentRecord``.

The load-bearing invariant: the record ``fingerprint`` IS the probe CONFIG hash
(``provenance.config_sha256`` — the treatment identity), NEVER the verdict string
(an observation). Same probe config + different verdict (PASS vs STOP) => SAME
fingerprint; different config => different fingerprint. Discovery records are not
run through ``compute_fingerprint`` — their fingerprint IS config_sha256 — so
these tests assert the adapter wires it through unchanged and never folds the
verdict in.
"""

from __future__ import annotations

import logging

import pytest

from discovery_verdict import VerdictProvenance
from discovery_verdict.builder import build_verdict

from hft_ops.ledger.dedup import FingerprintNormalizationError
from hft_ops.ledger.discovery_record import record_from_verdict

SHA_A = "a" * 64
SHA_B = "b" * 64

# The Phase-Y composer logs an expected/benign "missing components" WARN for
# every discovery record (it structurally lacks the 4 trust components).
logging.getLogger("hft_contracts.experiment_recorder").setLevel(logging.ERROR)


def _verdict(verdict_str: str, cfg_sha, *, edge=None):
    return build_verdict(
        probe_id="P_A9",
        study="my_probe",
        source_tree="glbx_discovery",
        verdict=verdict_str,
        any_tradeable_edge=edge,
        verdict_authority="DE-RISKING ONLY",
        provenance=VerdictProvenance(
            config_sha256=cfg_sha, seed=42, hft_metrics_version="0.1.26"
        ),
        honest_summary="summary text",
        hypothesis="hyp text",
        deflated_sharpe_ratio=0.79,
        dsr_classification="not_significant",
        selection_adjusted_significant=False,
        study_body={"n_days": 19},
    )


def test_fingerprint_is_config_sha_not_verdict(tmp_path):
    """Same probe CONFIG + DIFFERENT verdict string => SAME record fingerprint;
    the verdict is an OBSERVATION (training_metrics), never folded into the
    fingerprint."""
    r_pass = record_from_verdict(_verdict("PASS", SHA_A), pipeline_root=tmp_path)
    r_stop = record_from_verdict(_verdict("STOP", SHA_A), pipeline_root=tmp_path)

    assert r_pass.fingerprint == SHA_A
    assert r_stop.fingerprint == SHA_A
    assert r_pass.fingerprint == r_stop.fingerprint, (
        "verdict string must not change the fingerprint (it is config_sha256)"
    )
    # the verdict landed ONLY on the observation side
    assert r_pass.training_metrics["verdict"] == "PASS"
    assert r_stop.training_metrics["verdict"] == "STOP"


def test_different_config_different_fingerprint(tmp_path):
    r_a = record_from_verdict(_verdict("FAIL", SHA_A), pipeline_root=tmp_path)
    r_b = record_from_verdict(_verdict("FAIL", SHA_B), pipeline_root=tmp_path)
    assert r_a.fingerprint == SHA_A
    assert r_b.fingerprint == SHA_B
    assert r_a.fingerprint != r_b.fingerprint


def test_record_type_and_observation_mapping(tmp_path):
    r = record_from_verdict(_verdict("FAIL", SHA_A, edge=False), pipeline_root=tmp_path)
    assert r.record_type == "discovery"
    assert r.status == "completed"
    assert r.stages_completed == ["discovery"]
    assert r.name == "P_A9"
    assert r.notes == "summary text"
    assert r.hypothesis == "hyp text"
    assert "glbx_discovery" in r.tags
    assert "DE-RISKING ONLY" in r.tags
    # verdict + rails on the observation side
    assert r.training_metrics["verdict"] == "FAIL"
    assert r.training_metrics["any_tradeable_edge"] is False
    assert r.training_metrics["deflated_sharpe_ratio"] == pytest.approx(0.79)
    assert r.training_metrics["dsr_classification"] == "not_significant"
    # experiment_id = {probe}_{ts}_{fingerprint[:8]}
    assert r.experiment_id.startswith("P_A9_")
    assert r.experiment_id.endswith("_" + SHA_A[:8])


def test_verdict_never_enters_fingerprint_input(tmp_path):
    """Defense-in-depth: neither the verdict string nor any rail value appears in
    the fingerprint (which is exactly config_sha256, unchanged)."""
    r = record_from_verdict(_verdict("REAL_BUT_NOT_TRADEABLE", SHA_A), pipeline_root=tmp_path)
    assert r.fingerprint == SHA_A
    assert "REAL_BUT_NOT_TRADEABLE" not in r.fingerprint


def test_fail_loud_on_missing_config_sha(tmp_path):
    v = build_verdict(
        probe_id="P_A9",
        study="s",
        source_tree="glbx_discovery",
        verdict="FAIL",
        any_tradeable_edge=None,
        provenance=VerdictProvenance(config_sha256=None),
    )
    with pytest.raises(FingerprintNormalizationError):
        record_from_verdict(v, pipeline_root=tmp_path)


def test_fail_loud_on_malformed_config_sha(tmp_path):
    v = build_verdict(
        probe_id="P_A9",
        study="s",
        source_tree="glbx_discovery",
        verdict="FAIL",
        any_tradeable_edge=None,
        provenance=VerdictProvenance(config_sha256="not-a-64-hex-hash"),
    )
    with pytest.raises(FingerprintNormalizationError):
        record_from_verdict(v, pipeline_root=tmp_path)


def test_accepts_raw_dict_via_discovery_verdict_adapters(tmp_path):
    """A raw harness dict is normalized through the shared discovery_verdict
    adapters (reuse — no bespoke re-parse)."""
    raw = {
        "probe": "P_A9",
        "study": "s",
        "source_tree": "glbx_discovery",
        "verdict": "FAIL",
        "any_tradeable_edge": False,
        "provenance": {"config_sha256": SHA_A, "seed": 42},
    }
    r = record_from_verdict(
        raw,
        pipeline_root=tmp_path,
        source_tree="glbx_discovery",
        source_path="/x/results/p.json",
    )
    assert r.fingerprint == SHA_A
    assert r.record_type == "discovery"
    assert r.training_metrics["verdict"] == "FAIL"


def test_ledger_path_persists_record(tmp_path):
    ledger = tmp_path / "ledger"
    (ledger / "records").mkdir(parents=True)
    r = record_from_verdict(
        _verdict("FAIL", SHA_A), pipeline_root=tmp_path, ledger_path=ledger
    )
    written = ledger / "records" / f"{r.experiment_id}.json"
    assert written.exists()
    from hft_contracts.experiment_record import ExperimentRecord

    loaded = ExperimentRecord.load(written)
    assert loaded.fingerprint == SHA_A
    assert loaded.record_type == "discovery"
    assert loaded.notes == "summary text"
