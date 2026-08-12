from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

import hft_ops.publication.feature_carrier_admission as admission_module
from hft_contracts.generation_resolver import (
    GenerationResolutionError,
    ResolvedFeatureCarrierAdmissionCandidateV1,
    TrustedFeatureCarrierBuildPolicyV1,
)
from hft_contracts.portable_identity import portable_sha256
from hft_ops.publication.feature_carrier_admission import (
    FeatureCarrierAdmissionIssueError,
    issue_feature_carrier_admission,
)


def _build(repository: str, digit: str, toolchain_id: str) -> dict:
    return {
        "repository": repository,
        "commit": digit * 40,
        "git_dirty": False,
        "binary_sha256": digit * 64,
        "dependency_lock_sha256": chr(ord(digit) + 1) * 64,
        "dependency_graph_sha256": chr(ord(digit) + 2) * 64,
        "toolchain_id": toolchain_id,
    }


@pytest.fixture(autouse=True)
def _exercise_future_enabled_admission_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issuer behavior tests run under an explicit future enabled release."""

    monkeypatch.setattr(
        admission_module, "FEATURE_CARRIER_PRODUCTION_ADMISSION_ENABLED", True
    )


def _fixture(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    TrustedFeatureCarrierBuildPolicyV1,
    ResolvedFeatureCarrierAdmissionCandidateV1,
]:
    generation_root = tmp_path / "carrier"
    hft_ops_root = tmp_path / "hft-ops"
    generation_id = "1" * 64
    generation_receipt = (
        generation_root / "generations" / generation_id / "generation_receipt.json"
    )
    acceptance_receipt = generation_root / "acceptance" / "acceptance.json"
    generation_receipt.parent.mkdir(parents=True)
    acceptance_receipt.parent.mkdir(parents=True)
    generation_receipt.write_text("generation\n")
    acceptance_receipt.write_text("acceptance\n")
    generation_sha = hashlib.sha256(generation_receipt.read_bytes()).hexdigest()
    acceptance_sha = hashlib.sha256(acceptance_receipt.read_bytes()).hexdigest()
    policy_content_sha = "2" * 64
    policy_path = (
        hft_ops_root
        / "ledger"
        / "mbo_backbone_build_trust"
        / f"{policy_content_sha}.json"
    )
    policy_path.parent.mkdir(parents=True)
    policy_path.write_text("trusted-policy\n")
    policy_file_sha = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    trust = TrustedFeatureCarrierBuildPolicyV1(path=policy_path, sha256=policy_file_sha)
    build_binding = {
        "policy_id": "3" * 64,
        "policy_content_sha256": policy_content_sha,
        "policy_file_sha256": policy_file_sha,
        "qualified_pair_id": "4" * 64,
    }
    acceptance_content = {
        "acceptance_id": "5" * 64,
        "generation_content_sha256": "6" * 64,
    }
    candidate = ResolvedFeatureCarrierAdmissionCandidateV1(
        root=generation_root,
        generation_id=generation_id,
        generation_receipt=generation_receipt,
        generation_receipt_sha256=generation_sha,
        acceptance_receipt=acceptance_receipt,
        acceptance_receipt_sha256=acceptance_sha,
        global_artifacts={},
        partition_artifacts={},
        generation_content={"generation_id": generation_id},
        acceptance_content=acceptance_content,
        acceptance_evidence={},
        acceptance_qualifications={},
        acceptance_proven_gates=frozenset({"independent_gate"}),
        build_trust_policy={
            "policy_id": build_binding["policy_id"],
            "authority": _authority(),
        },
        build_trust_binding=build_binding,
        qualified_build_pair_id=build_binding["qualified_pair_id"],
    )
    return generation_root, hft_ops_root, trust, candidate


def _authority() -> dict:
    return _build("nagarx/hft-ops", "7", "python3_14_2_cpython")


def test_exact_candidate_issues_one_immutable_idempotent_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generation_root, hft_ops_root, trust, candidate = _fixture(tmp_path)
    monkeypatch.setattr(
        admission_module,
        "resolve_feature_carrier_admission_candidate",
        lambda *args, **kwargs: candidate,
    )
    monkeypatch.setattr(admission_module, "_authority_identity", lambda _: _authority())
    first = issue_feature_carrier_admission(
        generation_root=generation_root,
        acceptance_receipt="acceptance/acceptance.json",
        hft_ops_root=hft_ops_root,
        trusted_build_policy=trust,
        admitted_at_utc="2026-08-04T14:00:00Z",
    )
    second = issue_feature_carrier_admission(
        generation_root=generation_root,
        acceptance_receipt="acceptance/acceptance.json",
        hft_ops_root=hft_ops_root,
        trusted_build_policy=trust,
        admitted_at_utc="2026-08-04T14:00:00Z",
    )
    assert first.path == second.path
    assert first.write_result.created is True
    assert second.write_result.created is False
    assert first.path.name == f"{first.admission_id}.json"
    assert first.file_sha256 == second.file_sha256


def test_development_contract_blocks_issuer_before_root_or_candidate_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hft_contracts._generated import (
        FEATURE_CARRIER_PRODUCTION_ADMISSION_ENABLED as GENERATED_ENABLED,
    )

    assert GENERATED_ENABLED is False
    monkeypatch.setattr(
        admission_module, "FEATURE_CARRIER_PRODUCTION_ADMISSION_ENABLED", False
    )

    def unexpected(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("disabled issuer must not inspect a candidate")

    monkeypatch.setattr(
        admission_module, "resolve_feature_carrier_admission_candidate", unexpected
    )
    with pytest.raises(
        FeatureCarrierAdmissionIssueError, match="admission is disabled"
    ):
        issue_feature_carrier_admission(
            generation_root=Path("/private/tmp/nonexistent-generation-root"),
            acceptance_receipt="acceptance.json",
            hft_ops_root=Path("/private/tmp/nonexistent-hft-ops-root"),
            trusted_build_policy=TrustedFeatureCarrierBuildPolicyV1(
                path=Path("/private/tmp/nonexistent-policy.json"), sha256="a" * 64
            ),
            admitted_at_utc="2026-08-04T14:00:00Z",
        )


def test_candidate_change_blocks_before_admission_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generation_root, hft_ops_root, trust, candidate = _fixture(tmp_path)
    changed = copy.deepcopy(candidate)
    object.__setattr__(changed, "acceptance_receipt_sha256", "f" * 64)
    results = iter((candidate, changed))
    monkeypatch.setattr(
        admission_module,
        "resolve_feature_carrier_admission_candidate",
        lambda *args, **kwargs: next(results),
    )
    monkeypatch.setattr(admission_module, "_authority_identity", lambda _: _authority())
    with pytest.raises(FeatureCarrierAdmissionIssueError, match="changed"):
        issue_feature_carrier_admission(
            generation_root=generation_root,
            acceptance_receipt="acceptance/acceptance.json",
            hft_ops_root=hft_ops_root,
            trusted_build_policy=trust,
            admitted_at_utc="2026-08-04T14:00:00Z",
        )
    assert not (generation_root / "admissions").exists()


def test_policy_outside_hft_ops_root_blocks_before_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generation_root, hft_ops_root, trust, _ = _fixture(tmp_path)
    escaped = tmp_path / "outside-policy.json"
    escaped.write_bytes(trust.path.read_bytes())
    escaped_trust = TrustedFeatureCarrierBuildPolicyV1(
        path=escaped,
        sha256=hashlib.sha256(escaped.read_bytes()).hexdigest(),
    )
    called = False

    def unexpected(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("candidate resolver must not run")

    monkeypatch.setattr(
        admission_module, "resolve_feature_carrier_admission_candidate", unexpected
    )
    with pytest.raises(FeatureCarrierAdmissionIssueError, match="authority root"):
        issue_feature_carrier_admission(
            generation_root=generation_root,
            acceptance_receipt="acceptance/acceptance.json",
            hft_ops_root=hft_ops_root,
            trusted_build_policy=escaped_trust,
            admitted_at_utc="2026-08-04T14:00:00Z",
        )
    assert called is False
    assert not (generation_root / "admissions").exists()


def test_noncanonical_policy_ledger_path_blocks_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generation_root, hft_ops_root, trust, candidate = _fixture(tmp_path)
    wrong = hft_ops_root / "ledger" / "mbo_backbone_build_trust" / "wrong.json"
    wrong.write_bytes(trust.path.read_bytes())
    wrong_trust = TrustedFeatureCarrierBuildPolicyV1(
        path=wrong, sha256=hashlib.sha256(wrong.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        admission_module,
        "resolve_feature_carrier_admission_candidate",
        lambda *args, **kwargs: candidate,
    )
    with pytest.raises(FeatureCarrierAdmissionIssueError, match="canonical"):
        issue_feature_carrier_admission(
            generation_root=generation_root,
            acceptance_receipt="acceptance/acceptance.json",
            hft_ops_root=hft_ops_root,
            trusted_build_policy=wrong_trust,
            admitted_at_utc="2026-08-04T14:00:00Z",
        )
    assert not (generation_root / "admissions").exists()


def test_acceptance_content_hash_is_derived_not_caller_supplied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generation_root, hft_ops_root, trust, candidate = _fixture(tmp_path)
    monkeypatch.setattr(
        admission_module,
        "resolve_feature_carrier_admission_candidate",
        lambda *args, **kwargs: candidate,
    )
    monkeypatch.setattr(admission_module, "_authority_identity", lambda _: _authority())
    issued = issue_feature_carrier_admission(
        generation_root=generation_root,
        acceptance_receipt="acceptance/acceptance.json",
        hft_ops_root=hft_ops_root,
        trusted_build_policy=trust,
        admitted_at_utc="2026-08-04T14:00:00Z",
    )
    import json

    envelope = json.loads(issued.path.read_text())
    assert envelope["content"]["acceptance_content_sha256"] == portable_sha256(
        candidate.acceptance_content
    )


def test_policy_authority_must_equal_executing_hft_ops_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generation_root, hft_ops_root, trust, candidate = _fixture(tmp_path)
    untrusted = copy.deepcopy(candidate)
    object.__setattr__(
        untrusted,
        "build_trust_policy",
        {
            **candidate.build_trust_policy,
            "authority": _build("nagarx/hft-ops", "8", "python3_14_2_cpython"),
        },
    )
    monkeypatch.setattr(
        admission_module,
        "resolve_feature_carrier_admission_candidate",
        lambda *args, **kwargs: untrusted,
    )
    monkeypatch.setattr(admission_module, "_authority_identity", lambda _: _authority())
    with pytest.raises(FeatureCarrierAdmissionIssueError, match="policy authority"):
        issue_feature_carrier_admission(
            generation_root=generation_root,
            acceptance_receipt="acceptance/acceptance.json",
            hft_ops_root=hft_ops_root,
            trusted_build_policy=trust,
            admitted_at_utc="2026-08-04T14:00:00Z",
        )
    assert not (generation_root / "admissions").exists()


def test_final_candidate_change_blocks_before_admission_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generation_root, hft_ops_root, trust, candidate = _fixture(tmp_path)
    calls = 0

    def resolve_then_corrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise GenerationResolutionError("artifact bytes changed")
        return candidate

    monkeypatch.setattr(
        admission_module,
        "resolve_feature_carrier_admission_candidate",
        resolve_then_corrupt,
    )
    monkeypatch.setattr(admission_module, "_authority_identity", lambda _: _authority())
    with pytest.raises(FeatureCarrierAdmissionIssueError, match="before publication"):
        issue_feature_carrier_admission(
            generation_root=generation_root,
            acceptance_receipt="acceptance/acceptance.json",
            hft_ops_root=hft_ops_root,
            trusted_build_policy=trust,
            admitted_at_utc="2026-08-04T14:00:00Z",
        )
    assert not (generation_root / "admissions").exists()


def test_symlinked_admission_parent_cannot_escape_generation_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generation_root, hft_ops_root, trust, candidate = _fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (generation_root / "admissions").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(
        admission_module,
        "resolve_feature_carrier_admission_candidate",
        lambda *args, **kwargs: candidate,
    )
    monkeypatch.setattr(admission_module, "_authority_identity", lambda _: _authority())
    with pytest.raises(FeatureCarrierAdmissionIssueError, match="publication failed"):
        issue_feature_carrier_admission(
            generation_root=generation_root,
            acceptance_receipt="acceptance/acceptance.json",
            hft_ops_root=hft_ops_root,
            trusted_build_policy=trust,
            admitted_at_utc="2026-08-04T14:00:00Z",
        )
    assert list(outside.iterdir()) == []


def test_authority_change_after_candidate_validation_blocks_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generation_root, hft_ops_root, trust, candidate = _fixture(tmp_path)
    monkeypatch.setattr(
        admission_module,
        "resolve_feature_carrier_admission_candidate",
        lambda *args, **kwargs: candidate,
    )
    changed_authority = {
        **_authority(),
        "dependency_lock_sha256": "f" * 64,
    }
    identities = iter((_authority(), changed_authority))
    monkeypatch.setattr(
        admission_module, "_authority_identity", lambda _: next(identities)
    )
    with pytest.raises(FeatureCarrierAdmissionIssueError, match="authority changed"):
        issue_feature_carrier_admission(
            generation_root=generation_root,
            acceptance_receipt="acceptance/acceptance.json",
            hft_ops_root=hft_ops_root,
            trusted_build_policy=trust,
            admitted_at_utc="2026-08-04T14:00:00Z",
        )
    assert not (generation_root / "admissions").exists()


def test_claimed_authority_root_must_contain_the_executing_issuer(
    tmp_path: Path,
) -> None:
    counterfeit = tmp_path / "counterfeit-hft-ops"
    counterfeit.mkdir()
    with pytest.raises(
        FeatureCarrierAdmissionIssueError, match="executing admission issuer"
    ):
        admission_module._tracked_hft_ops_source_sha256(counterfeit)
