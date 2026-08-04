"""Issue a bounded, content-addressed schema-4 development authority.

The issuer inspects clean Git checkouts and committed evidence directly. It
never accepts caller-supplied repository SHAs, evidence hashes, or permission
maps. The resulting receipt permits candidate code and quarantined D0/D1
evidence only; every production, admission, activation, history, and research
permission remains false in the hft-contracts schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from hft_contracts.atomic_io import ImmutableWriteResult, immutable_write_json
from hft_contracts.canonical_hash import canonical_json_blob, sha256_hex
from hft_contracts.publication_receipts import (
    PublicationContractError,
    build_hashed_envelope,
    validate_schema4_development_authority,
)


OPERATOR_DECISION_ID = "operator-message-2026-08-04-next-development-cycle"
PACKET_RELATIVE = Path("contracts/mbo_backbone/packet_manifest_v1.json")
REMEDIATION_RELATIVE = Path("contracts/mbo_backbone/remediation_authority_v1.json")
D0_RELATIVE = Path("contracts/mbo_backbone/d0_evidence_receipt_v2.json")
AUTHORITY_LEDGER_RELATIVE = Path("ledger/mbo_backbone_transitions/schema4_development")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AuthorityIssueError(RuntimeError):
    """The issuer cannot prove the inputs needed for bounded authority."""


@dataclass(frozen=True)
class IssuedDevelopmentAuthority:
    path: Path
    content_sha256: str
    file_sha256: str
    write_result: ImmutableWriteResult


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AuthorityIssueError(
            f"git {' '.join(arguments)} failed in {repository}: "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _canonical_github_repository(remote: str) -> str | None:
    value = remote.strip()
    if value.startswith("git@github.com:"):
        path = value.removeprefix("git@github.com:")
    else:
        parsed = urlparse(value)
        if parsed.hostname != "github.com" or parsed.scheme not in {"https", "ssh"}:
            return None
        path = parsed.path.lstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", path) is None:
        return None
    return path


def _require_origin(repository: Path, expected: str) -> None:
    actual = _canonical_github_repository(
        _git(repository, "remote", "get-url", "origin")
    )
    if actual != expected:
        raise AuthorityIssueError(
            f"repository origin mismatch for {repository}: {actual!r} != {expected!r}"
        )


def _head(repository: Path) -> str:
    value = _git(repository, "rev-parse", "HEAD")
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise AuthorityIssueError(f"repository HEAD is not exact 40-hex: {repository}")
    return value


def _status_paths(repository: Path) -> set[str]:
    lines = _git(repository, "status", "--porcelain", "--untracked-files=all")
    if not lines:
        return set()
    paths: set[str] = set()
    for line in lines.splitlines():
        if len(line) < 4:
            raise AuthorityIssueError(
                f"malformed git status line in {repository}: {line!r}"
            )
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.add(path)
    return paths


def _require_clean(repository: Path, *, label: str) -> None:
    dirty = _status_paths(repository)
    if dirty:
        raise AuthorityIssueError(f"{label} checkout is dirty: {sorted(dirty)}")


def _stable_read(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise AuthorityIssueError(f"evidence is absent or a symlink: {path}")
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        payload = handle.read()
        after = os.fstat(handle.fileno())
    live = path.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    live_identity = (
        live.st_dev,
        live.st_ino,
        live.st_size,
        live.st_mtime_ns,
        live.st_ctime_ns,
    )
    if before_identity != after_identity or after_identity != live_identity:
        raise AuthorityIssueError(
            f"evidence changed or was replaced while read: {path}"
        )
    return payload


def _load_hashed_envelope(path: Path) -> tuple[dict, bytes]:
    payload = _stable_read(path)

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise AuthorityIssueError(f"duplicate JSON key in {path}: {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise AuthorityIssueError(f"non-finite JSON token in {path}: {value}")

    try:
        envelope = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthorityIssueError(f"invalid strict JSON in {path}: {exc}") from exc
    if not isinstance(envelope, dict) or set(envelope) != {
        "schema",
        "content",
        "content_sha256",
    }:
        raise AuthorityIssueError(f"non-exact hashed envelope: {path}")
    content = envelope.get("content")
    content_hash = envelope.get("content_sha256")
    if not isinstance(content, dict) or not isinstance(content_hash, str):
        raise AuthorityIssueError(f"malformed hashed envelope: {path}")
    if content_hash != sha256_hex(canonical_json_blob(content)):
        raise AuthorityIssueError(f"content hash mismatch: {path}")
    return envelope, payload


def _validate_evidence(root: Path) -> tuple[dict, dict, bytes]:
    packet, _ = _load_hashed_envelope(root / PACKET_RELATIVE)
    remediation, _ = _load_hashed_envelope(root / REMEDIATION_RELATIVE)
    d0, d0_payload = _load_hashed_envelope(root / D0_RELATIVE)
    packet_content = packet["content"]
    remediation_content = remediation["content"]
    d0_content = d0["content"]
    if (
        packet_content.get("artifact") != "mbo_backbone_packet_manifest_v1"
        or packet_content.get("status") != "not_admitted"
        or packet_content.get("external_admission_anchor") is not None
    ):
        raise AuthorityIssueError("historical packet is not explicitly non-admitted")
    selection = remediation_content.get("operator_selection_evidence")
    if (
        not isinstance(selection, dict)
        or selection.get("semantic_or_artifact_authorization") is not False
    ):
        raise AuthorityIssueError(
            "historical remediation record is not mechanical-only"
        )
    if (
        d0_content.get("artifact") != "mbo_d0_evidence_receipt_v2"
        or d0_content.get("status") != "observed_pass_nonadmitting"
        or d0_content.get("authorizes") != "nothing"
    ):
        raise AuthorityIssueError("D0 receipt is not corrected quarantined evidence")
    return packet, remediation, d0_payload


def issue_schema4_development_authority(
    *,
    pipeline_root: Path,
    hft_ops_root: Path,
    issued_at_utc: str,
) -> IssuedDevelopmentAuthority:
    pipeline_root = pipeline_root.resolve()
    hft_ops_root = hft_ops_root.resolve()
    contracts_root = (pipeline_root / "hft-contracts").resolve()
    wiki_root = (pipeline_root / "hft-wiki").resolve()

    for repository, expected in (
        (pipeline_root, "nagarx/hft-discovery"),
        (contracts_root, "nagarx/hft-contracts"),
        (wiki_root, "nagarx/hft-wiki"),
        (hft_ops_root, "nagarx/hft-ops"),
    ):
        _require_origin(repository, expected)
    _require_clean(pipeline_root, label="root evidence")
    _require_clean(contracts_root, label="hft-contracts validator")

    packet, remediation, d0_payload = _validate_evidence(pipeline_root)
    content = {
        "artifact": "mbo_schema4_development_authority_v1",
        "transition": "schema4_development_authorize",
        "status": "authorized_development_nonadmitting",
        "issuer": "hft-ops",
        "issued_at_utc": issued_at_utc,
        "operator_decision_id": OPERATOR_DECISION_ID,
        "authority_repository": "nagarx/hft-ops",
        "packet_manifest_content_sha256": packet["content_sha256"],
        "remediation_authority_content_sha256": remediation["content_sha256"],
        "development_evidence": {
            "d0_receipt": {
                "path": D0_RELATIVE.as_posix(),
                "sha256": hashlib.sha256(d0_payload).hexdigest(),
            }
        },
        "repository_bindings": {
            "root": {
                "repository": "nagarx/hft-discovery",
                "commit": _head(pipeline_root),
            },
            "hft_contracts": {
                "repository": "nagarx/hft-contracts",
                "commit": _head(contracts_root),
            },
            "hft_ops": {
                "repository": "nagarx/hft-ops",
                "baseline_commit": _head(hft_ops_root),
            },
            "hft_wiki": {
                "repository": "nagarx/hft-wiki",
                "evidence_commit": _head(wiki_root),
            },
        },
        "supersession": {
            "supersedes_local_restriction": "phase_0_mechanical_only",
            "basis": "operator_message_2026_08_04_next_development_cycle",
            "does_not_assert_phase0_closure": True,
        },
        "implementation_permissions": {
            "schema4_contract_and_codegen": True,
            "shared_receipt_contracts": True,
            "generation_resolver": True,
            "hft_ops_admission_activation_code": True,
            "reconstructor_v0_4_candidate_code": True,
            "extractor_schema4_candidate_code": True,
        },
        "artifact_permissions": {
            "quarantined_d0_d1_evidence": True,
            "corrected_market_artifact_generation": False,
            "production_generation": False,
            "generation_admission": False,
            "generation_activation": False,
            "historical_rederivation": False,
            "historical_result_admission": False,
        },
        "claim_permissions": {"research_claims": False, "generic_f064": False},
    }
    envelope = build_hashed_envelope("mbo_schema4_development_authority_v1", content)
    try:
        validate_schema4_development_authority(envelope)
    except PublicationContractError as exc:
        raise AuthorityIssueError(
            f"generated authority violates its contract: {exc}"
        ) from exc

    content_hash = envelope["content_sha256"]
    if not isinstance(content_hash, str) or SHA256_RE.fullmatch(content_hash) is None:
        raise AuthorityIssueError("generated authority content hash is malformed")
    target_relative = AUTHORITY_LEDGER_RELATIVE / f"{content_hash}.json"
    dirty = _status_paths(hft_ops_root)
    if dirty - {target_relative.as_posix()}:
        raise AuthorityIssueError(
            f"hft-ops checkout has unrelated changes: {sorted(dirty)}"
        )
    target = hft_ops_root / target_relative
    result = immutable_write_json(target, envelope)
    return IssuedDevelopmentAuthority(
        path=target,
        content_sha256=content_hash,
        file_sha256=result.sha256,
        write_result=result,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument(
        "--hft-ops-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument("--issued-at-utc", required=True)
    args = parser.parse_args()
    try:
        issued = issue_schema4_development_authority(
            pipeline_root=args.pipeline_root,
            hft_ops_root=args.hft_ops_root,
            issued_at_utc=args.issued_at_utc,
        )
    except (AuthorityIssueError, OSError, ValueError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "path": str(issued.path),
                "content_sha256": issued.content_sha256,
                "file_sha256": issued.file_sha256,
                "created": issued.write_result.created,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
