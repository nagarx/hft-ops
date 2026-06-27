"""record_from_verdict — adapt a normalized discovery ``Verdict`` into a
first-class, fingerprinted ``ExperimentRecord`` so each intraday discovery probe
is comparable / dedup'd / monitorable alongside training runs in the experiment
ledger.

Design (per hft-rules §0 reuse-first / §1 single-source-of-truth):

  * **Record construction delegates to the SSoT.** We never hand-roll an
    ``ExperimentRecord``; we call
    ``hft_contracts.experiment_recorder.record_from_artifacts`` (the single
    ExperimentRecord construction site, shared with the hft-ops orchestrator +
    the lob-model-trainer direct path). ``record_type="discovery"``
    (``hft_contracts >= 2.10.0``).

  * **Verdict parsing reuses ``discovery_verdict``.** A raw harness dict is
    normalized via ``discovery_verdict.normalize_verdict`` + the shared
    ``KNOWN_ADAPTERS`` (same machinery the F5 monitor's ``DiscoveryVerdictReader``
    uses) — we do NOT re-parse the JSON shapes by hand.

  * **The fingerprint IS the probe CONFIG hash.** ``provenance.config_sha256``
    is the *treatment identity* of a probe — the same probe config run twice
    (e.g. before/after a verdict flips PASS→STOP) is the SAME experiment. We
    FAIL-LOUD (``FingerprintNormalizationError``) if it is missing / not a
    64-hex SHA-256, mirroring the dedup module's hard-fail policy — a degenerate
    fingerprint would re-introduce the ledger-conflation class (dedup Phase
    3 §3.3b). The verdict string is an OBSERVATION; it lands ONLY on the
    observation side (``training_metrics`` / ``notes``) and NEVER enters any
    fingerprint input.

  * **Phase Y provenance is structurally absent.** A discovery probe has none
    of the 4 trust components (data_export_fp / feature_set_content_hash /
    compatibility_fp / model_config_hash), so we pass
    ``require_complete_provenance=False`` — the composer's "skipped — missing
    ..." WARN from ``record_from_artifacts`` is expected and benign for this
    record type.

Read-only with respect to the harnesses: this maps an already-produced verdict;
it imports no harness code.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, Union

from discovery_verdict import KNOWN_ADAPTERS, Verdict, normalize_verdict
from hft_contracts import SCHEMA_VERSION
from hft_contracts.experiment_recorder import record_from_artifacts
from hft_contracts.experiment_record import ExperimentRecord, RecordType
from hft_contracts.signal_manifest import CONTENT_HASH_RE

from hft_ops.ledger.dedup import FingerprintNormalizationError

__all__ = ["record_from_verdict"]


def _coerce_verdict(
    verdict_or_dict: Union[Verdict, Mapping[str, Any]],
    *,
    source_path: str,
    source_tree: str,
) -> Verdict:
    """Return a normalized :class:`Verdict`, reusing the ``discovery_verdict``
    adapters when a raw harness dict is passed (no bespoke JSON re-parse)."""
    if isinstance(verdict_or_dict, Verdict):
        return verdict_or_dict
    if isinstance(verdict_or_dict, Mapping):
        tree = source_tree or str(verdict_or_dict.get("source_tree") or "")
        if not tree:
            raise ValueError(
                "record_from_verdict: a raw verdict dict needs a source_tree "
                "(pass source_tree=... or include a 'source_tree' key) so the "
                "discovery_verdict adapters can normalize it."
            )
        return normalize_verdict(
            verdict_or_dict,
            source_path=source_path,
            source_tree=tree,
            adapters=KNOWN_ADAPTERS,
        )
    raise TypeError(
        "record_from_verdict: expected a discovery_verdict.Verdict or a raw "
        f"verdict dict, got {type(verdict_or_dict).__name__}."
    )


def record_from_verdict(
    verdict_or_dict: Union[Verdict, Mapping[str, Any]],
    *,
    pipeline_root: Union[Path, str],
    contract_version: Optional[str] = None,
    source_path: str = "",
    source_tree: str = "",
    ledger_path: Optional[Union[Path, str]] = None,
) -> ExperimentRecord:
    """Map a discovery ``Verdict`` to a fingerprinted :class:`ExperimentRecord`.

    Args:
        verdict_or_dict: A normalized :class:`discovery_verdict.Verdict`, OR a
            raw harness ``results/*.json`` dict (normalized here via the shared
            ``discovery_verdict`` adapters — reuse, no re-parse).
        pipeline_root: Monorepo root, passed through to ``build_provenance`` for
            git capture (a discovery probe carries no manifest/config paths, so
            provenance is git-only — fail-open on a non-git tree).
        contract_version: Pipeline schema version stamped on the record.
            Defaults to ``hft_contracts.SCHEMA_VERSION``.
        source_path / source_tree: Lineage for a raw-dict input (ignored when a
            ``Verdict`` is passed — it already carries them).
        ledger_path: If set, atomic-write the composed record to
            ``<ledger_path>/records/<experiment_id>.json`` (via the
            :meth:`ExperimentRecord.save` ``atomic_write_json`` SSoT). Must be an
            existing directory — fail-loud otherwise (no silent auto-create, per
            hft-rules §5).

    Returns:
        The composed :class:`ExperimentRecord` with ``record_type="discovery"``,
        ``fingerprint = verdict.provenance.config_sha256`` (the treatment
        identity), and the verdict mapped onto the observation side
        (``training_metrics`` + ``notes``).

    Raises:
        FingerprintNormalizationError: If ``provenance.config_sha256`` is missing
            or not a 64-hex lowercase SHA-256 — we never fingerprint a degenerate
            value (would re-introduce the dedup ledger-conflation class).
        TypeError / ValueError: On an unusable input shape (see
            :func:`_coerce_verdict`).
    """
    verdict = _coerce_verdict(
        verdict_or_dict, source_path=source_path, source_tree=source_tree
    )

    # The probe CONFIG hash is the treatment identity → the record fingerprint.
    # FAIL-LOUD on a degenerate value (mirrors the dedup module's HARD-FAIL
    # policy; a None/short/uppercase value would silently conflate distinct
    # probes in the ledger — dedup Phase 3 §3.3b recurrence).
    config_sha = verdict.provenance.config_sha256
    if not isinstance(config_sha, str) or not CONTENT_HASH_RE.match(config_sha):
        raise FingerprintNormalizationError(
            "record_from_verdict: cannot fingerprint discovery probe "
            f"{verdict.probe_id!r} — provenance.config_sha256 must be a 64-hex "
            f"lowercase SHA-256 (the probe CONFIG hash is the treatment "
            f"identity); got {config_sha!r}. Emit the verdict via "
            f"discovery_verdict.build_verdict with "
            f"VerdictProvenance(config_sha256=<resolved hash>)."
        )

    # OBSERVATION dict — the verdict + its rails. These keys are NOT in the
    # index_entry() training-metrics whitelist (they stay in the record body,
    # per the `analysis` convention); the F5 monitor surfaces the verdict string
    # by collapsing this record with its on-disk discovery verdict row. NONE of
    # these enter the fingerprint.
    training_metrics: dict[str, Any] = {
        "verdict": verdict.verdict,
        "any_tradeable_edge": verdict.any_tradeable_edge,
        "acquisition_decision": verdict.acquisition_decision,
        "power_class": verdict.power_class,
        "mde": verdict.mde,
        "family_fwer_p": verdict.family_fwer_p,
    }
    if verdict.deflated_sharpe_ratio is not None:
        training_metrics["deflated_sharpe_ratio"] = verdict.deflated_sharpe_ratio
    if verdict.dsr_classification is not None:
        training_metrics["dsr_classification"] = verdict.dsr_classification
    if verdict.selection_adjusted_significant is not None:
        training_metrics["selection_adjusted_significant"] = (
            verdict.selection_adjusted_significant
        )

    # Delegate construction to the hft-contracts SSoT (single ExperimentRecord
    # build site). ledger_path is handled here (not delegated) because we set
    # `notes` AFTER construction — record_from_artifacts has no notes parameter,
    # and it would persist before we could attach the honest summary.
    record = record_from_artifacts(
        name=verdict.probe_id,
        pipeline_root=Path(pipeline_root),
        contract_version=contract_version or str(SCHEMA_VERSION),
        fingerprint=config_sha,
        training_metrics=training_metrics,
        tags=[t for t in (verdict.source_tree, verdict.verdict_authority) if t],
        hypothesis=verdict.hypothesis or "",
        record_type=RecordType.DISCOVERY.value,
        status="completed",
        stages_completed=["discovery"],
        # A discovery probe structurally lacks the 4 Phase Y trust components;
        # the composer's missing-components WARN is expected/benign here.
        require_complete_provenance=False,
    )

    # `notes` is a post-construction observation field (mutable per the
    # ExperimentRecord docstring); record_from_artifacts has no notes kwarg.
    record.notes = verdict.honest_summary or ""

    if ledger_path is not None:
        ledger_path = Path(ledger_path)
        if not ledger_path.exists():
            raise ValueError(
                f"record_from_verdict: ledger_path {ledger_path} does not exist. "
                f"Caller must provide an existing directory; this adapter does "
                f"NOT auto-create the parent (would hide operator typos per "
                f"hft-rules §5)."
            )
        records_dir = ledger_path / "records"
        records_dir.mkdir(parents=True, exist_ok=True)
        # ExperimentRecord.save() uses the atomic_write_json SSoT internally.
        record.save(records_dir / f"{record.experiment_id}.json")

    return record
