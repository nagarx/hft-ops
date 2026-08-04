from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hft_contracts.publication_receipts import (
    build_hashed_envelope,
    validate_schema4_development_authority,
)
from hft_ops.publication.development_authority import (
    AuthorityIssueError,
    issue_schema4_development_authority,
)


ISSUED_AT = "2026-08-04T12:00:00Z"


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _init(repository: Path, origin: str) -> None:
    repository.mkdir(parents=True)
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Authority Test")
    _git(repository, "config", "user.email", "authority@example.invalid")
    _git(repository, "remote", "add", "origin", f"https://github.com/{origin}.git")


def _commit(repository: Path, message: str) -> str:
    _git(repository, "add", ".")
    _git(repository, "commit", "-q", "-m", message)
    return _git(repository, "rev-parse", "HEAD")


def _write_envelope(path: Path, schema: str, content: dict) -> None:
    envelope = build_hashed_envelope(schema, content)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _repositories(tmp_path: Path, *, bad_d0: bool = False) -> tuple[Path, Path]:
    root = tmp_path / "root"
    ops = tmp_path / "ops"
    _init(root, "nagarx/hft-discovery")
    (root / ".gitignore").write_text("hft-contracts/\nhft-wiki/\n", encoding="utf-8")

    contracts = root / "hft-contracts"
    wiki = root / "hft-wiki"
    _init(contracts, "nagarx/hft-contracts")
    _init(wiki, "nagarx/hft-wiki")
    _init(ops, "nagarx/hft-ops")
    (contracts / "validator.txt").write_text("validator\n", encoding="utf-8")
    (wiki / "evidence.txt").write_text("evidence\n", encoding="utf-8")
    (ops / "code.txt").write_text("authority issuer\n", encoding="utf-8")
    _commit(contracts, "validator")
    _commit(wiki, "evidence")
    _commit(ops, "issuer")

    _write_envelope(
        root / "contracts/mbo_backbone/packet_manifest_v1.json",
        "mbo_backbone_packet_manifest_v1",
        {
            "artifact": "mbo_backbone_packet_manifest_v1",
            "status": "not_admitted",
            "external_admission_anchor": None,
        },
    )
    _write_envelope(
        root / "contracts/mbo_backbone/remediation_authority_v1.json",
        "mbo_backbone_hashed_envelope_v1",
        {
            "artifact": "mbo_backbone_remediation_authority_v1",
            "operator_selection_evidence": {
                "semantic_or_artifact_authorization": False,
            },
        },
    )
    _write_envelope(
        root / "contracts/mbo_backbone/d0_evidence_receipt_v2.json",
        "mbo_backbone_hashed_envelope_v1",
        {
            "artifact": "mbo_d0_evidence_receipt_v2",
            "status": "invalid" if bad_d0 else "observed_pass_nonadmitting",
            "authorizes": "nothing",
        },
    )
    _commit(root, "development evidence")
    return root, ops


def test_issuer_derives_exact_bindings_and_is_idempotent(tmp_path: Path) -> None:
    root, ops = _repositories(tmp_path)
    first = issue_schema4_development_authority(
        pipeline_root=root,
        hft_ops_root=ops,
        issued_at_utc=ISSUED_AT,
    )
    assert first.write_result.created is True
    envelope = json.loads(first.path.read_text(encoding="utf-8"))
    content = validate_schema4_development_authority(envelope)
    assert content["repository_bindings"]["root"]["commit"] == _git(
        root, "rev-parse", "HEAD"
    )
    assert content["artifact_permissions"]["production_generation"] is False
    assert content["claim_permissions"]["research_claims"] is False

    second = issue_schema4_development_authority(
        pipeline_root=root,
        hft_ops_root=ops,
        issued_at_utc=ISSUED_AT,
    )
    assert second.path == first.path
    assert second.file_sha256 == first.file_sha256
    assert second.write_result.created is False


def test_issuer_rejects_dirty_identity_input(tmp_path: Path) -> None:
    root, ops = _repositories(tmp_path)
    (root / "untracked.txt").write_text("drift\n", encoding="utf-8")
    with pytest.raises(AuthorityIssueError, match="root evidence checkout is dirty"):
        issue_schema4_development_authority(
            pipeline_root=root,
            hft_ops_root=ops,
            issued_at_utc=ISSUED_AT,
        )


def test_issuer_rejects_nonadmissible_d0(tmp_path: Path) -> None:
    root, ops = _repositories(tmp_path, bad_d0=True)
    with pytest.raises(AuthorityIssueError, match="corrected quarantined evidence"):
        issue_schema4_development_authority(
            pipeline_root=root,
            hft_ops_root=ops,
            issued_at_utc=ISSUED_AT,
        )


def test_issuer_rejects_unrelated_hft_ops_change(tmp_path: Path) -> None:
    root, ops = _repositories(tmp_path)
    (ops / "unrelated.txt").write_text("hitchhike\n", encoding="utf-8")
    with pytest.raises(AuthorityIssueError, match="unrelated changes"):
        issue_schema4_development_authority(
            pipeline_root=root,
            hft_ops_root=ops,
            issued_at_utc=ISSUED_AT,
        )
