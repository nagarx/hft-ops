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

import json
import logging

import pytest
from hft_contracts.provenance import (
    ALLOW_UNTRACKED_SOURCE_ENV,
    NOT_GIT_TRACKED_SENTINEL,
    UntrackedSourceError,
    build_provenance,
)

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


def test_fingerprint_is_config_sha_not_verdict(tmp_path, exploratory_provenance_override):
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


def test_different_config_different_fingerprint(tmp_path, exploratory_provenance_override):
    r_a = record_from_verdict(_verdict("FAIL", SHA_A), pipeline_root=tmp_path)
    r_b = record_from_verdict(_verdict("FAIL", SHA_B), pipeline_root=tmp_path)
    assert r_a.fingerprint == SHA_A
    assert r_b.fingerprint == SHA_B
    assert r_a.fingerprint != r_b.fingerprint


def test_record_type_and_observation_mapping(tmp_path, exploratory_provenance_override):
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


def test_verdict_never_enters_fingerprint_input(tmp_path, exploratory_provenance_override):
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


def test_accepts_raw_dict_via_discovery_verdict_adapters(tmp_path, exploratory_provenance_override):
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


def test_ledger_path_persists_record(tmp_path, exploratory_provenance_override):
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


# ---------------------------------------------------------------------------
# NEGATIVE CONTROL — the guard the six tests above opt out of must still go RED
# ---------------------------------------------------------------------------


class TestUntrackedSourceGuardIsLive:
    """Prove the guard that ``exploratory_provenance_override`` disables is ARMED.

    Six tests in this file take a deliberate override (see the fixture's
    docstring in ``tests/conftest.py``) so that ``record_from_verdict`` returns
    a record instead of refusing. Without the controls below that override
    would be INDISTINGUISHABLE from having deleted the protection: the next
    reader would see a fully green file and no evidence the guard exists.

    THIS IS NOT A HYPOTHETICAL. Measured in the sibling repair
    (``lob-model-trainer`` ``b94f462``): with the production guard forced to
    ``if False and ...`` in a PYTHONPATH-shadowed copy, ALL 23 pre-existing
    tests — including the seven just repaired — stayed GREEN. Only the negative
    control saw it.

    hft-rules §1: *"AN INSTRUMENT THAT CANNOT GO RED IS NOT AN INSTRUMENT.
    Before trusting any guard green, build the failing state and watch it
    fail."* These arms build it.

    ⚠️ Each arm pins the CAUSE rather than merely observing a failure: the
    exception TYPE, the message naming its own remedy, and an EMPTY ledger
    directory (refused outright, not partially written). A bare
    ``pytest.raises(Exception)`` would be satisfied by a large family of
    unrelated breakages.
    """

    def test_build_provenance_itself_refuses_an_untracked_tree(
        self, tmp_path, monkeypatch,
    ):
        """ARM 1 — the guard at its own boundary.

        ``delenv`` rather than "assume unset": an operator who exported
        ``HFT_ALLOW_UNTRACKED_SOURCE=1`` in their shell must not silently
        change which arm this test is measuring.
        """
        monkeypatch.delenv(ALLOW_UNTRACKED_SOURCE_ENV, raising=False)

        with pytest.raises(UntrackedSourceError) as excinfo:
            build_provenance(tmp_path)

        # The message must name its own remedy — otherwise the next agent hits
        # this wall with no way out and edits the test instead of the config.
        assert ALLOW_UNTRACKED_SOURCE_ENV in str(excinfo.value), (
            "UntrackedSourceError must name the override env var so an "
            f"operator can act on it; got: {excinfo.value}"
        )

    def test_record_from_verdict_refuses_and_writes_nothing(
        self, tmp_path, monkeypatch,
    ):
        """ARM 2 — the guard reaching HFT-OPS, through the real call chain.

        ARM 1 alone would still pass if ``discovery_record`` started passing
        ``allow_untracked_source=True`` on every call. This arm binds the
        adapter: it asserts the refusal survives ``record_from_verdict`` →
        ``record_from_artifacts`` → ``build_provenance``, and that a refused
        record leaves NO partial file behind.
        """
        monkeypatch.delenv(ALLOW_UNTRACKED_SOURCE_ENV, raising=False)
        ledger = tmp_path / "ledger"
        (ledger / "records").mkdir(parents=True)

        with pytest.raises(UntrackedSourceError):
            record_from_verdict(
                _verdict("FAIL", SHA_A), pipeline_root=tmp_path, ledger_path=ledger
            )

        assert list((ledger / "records").iterdir()) == [], (
            "guard refused but a record was still written: "
            f"{list((ledger / 'records').iterdir())}"
        )

    def test_override_admits_the_same_call_and_is_recorded_in_the_json(
        self, tmp_path, exploratory_provenance_override,
    ):
        """ARM 3 — matched positive. Same inputs as ARM 2; override the ONLY change.

        Two things are proven here that neither arm proves alone:

        1. ARM 2's ledger directory WAS writable, so its refusal cannot be
           blamed on a bad path.
        2. Taking the hatch is VISIBLE in the artifact. hft-contracts records it
           at ``provenance.allow_untracked_source`` precisely so a run is
           un-re-derivable *by choice* rather than by accident; if a future
           refactor dropped that field, the six overriding tests above would go
           on passing in silence and only this assertion would go red.
        """
        ledger = tmp_path / "ledger"
        (ledger / "records").mkdir(parents=True)

        r = record_from_verdict(
            _verdict("FAIL", SHA_A), pipeline_root=tmp_path, ledger_path=ledger
        )
        written = ledger / "records" / f"{r.experiment_id}.json"
        assert written.exists()

        doc = json.loads(written.read_text())
        assert doc["provenance"]["allow_untracked_source"] is True, (
            "the override was taken but is not recorded in the emitted record — "
            "an un-re-derivable run would look identical to a tracked one"
        )
        assert doc["provenance"]["git"]["commit_hash"] == NOT_GIT_TRACKED_SENTINEL

    def test_a_non_truthy_env_value_does_not_disable_the_guard(
        self, tmp_path, monkeypatch,
    ):
        """BOUNDARY — ``HFT_ALLOW_UNTRACKED_SOURCE=0`` must NOT open the hatch.

        The env var is a correctness gate read from an unvalidated string. Set
        to ``"0"`` it is *present but false*; a naive ``os.environ.get(...)``
        truthiness check would read it as ON and disable the guard for anyone
        who thought they were turning it OFF.
        """
        monkeypatch.setenv(ALLOW_UNTRACKED_SOURCE_ENV, "0")

        with pytest.raises(UntrackedSourceError):
            record_from_verdict(_verdict("FAIL", SHA_A), pipeline_root=tmp_path)

    def test_the_fingerprint_invariant_holds_without_the_override_too(
        self, tmp_path, exploratory_provenance_override,
    ):
        """The invariant this FILE exists to protect, stated independently.

        ``test_verdict_never_enters_fingerprint_input`` and
        ``test_fingerprint_is_config_sha_not_verdict`` were RED from
        2026-08-14 to 2026-09-07, so root ``CLAUDE.md``'s standing rule — *"a
        gate outcome is an OBSERVATION, not a TREATMENT"* — had no live guard
        for three weeks. This arm re-states it at the strongest available
        point: the fingerprint is byte-equal to the config sha for FOUR
        different verdict strings, so no verdict value can perturb it.
        """
        fps = {
            v: record_from_verdict(
                _verdict(v, SHA_A), pipeline_root=tmp_path
            ).fingerprint
            for v in ("PASS", "STOP", "FAIL", "REAL_BUT_NOT_TRADEABLE")
        }
        assert set(fps.values()) == {SHA_A}, (
            "the verdict string perturbed the fingerprint — a gate OUTCOME "
            f"leaked into the treatment identity: {fps}"
        )
