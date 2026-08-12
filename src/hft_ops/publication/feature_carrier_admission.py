"""Fail-closed hft-ops issuer for schema-v4 feature-carrier admission.

This module implements admission code only.  It cannot create a build-trust
policy, qualify builds, activate a generation, or update CURRENT.  In the
absence of an externally authorized policy it has no successful path.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from hft_contracts._generated import FEATURE_CARRIER_PRODUCTION_ADMISSION_ENABLED
from hft_contracts.atomic_io import (
    ImmutablePublicationError,
    ImmutableWriteResult,
    immutable_write_json_beneath,
)
from hft_contracts.feature_carrier_receipts import (
    ADMISSION_POLICY_ID,
    ADMISSION_SCHEMA,
    compute_feature_carrier_admission_id,
    validate_feature_carrier_admission_receipt,
)
from hft_contracts.generation_resolver import (
    GenerationResolutionError,
    ResolvedFeatureCarrierAdmissionCandidateV1,
    TrustedFeatureCarrierBuildPolicyV1,
    resolve_feature_carrier_admission_candidate,
)
from hft_contracts.portable_identity import (
    PORTABLE_CANONICALIZATION_ID,
    build_portable_envelope,
    portable_sha256,
)


BUILD_TRUST_LEDGER_RELATIVE = Path("ledger/mbo_backbone_build_trust")
ADMISSION_DIRECTORY = "admissions"
HFT_OPS_REPOSITORY = "nagarx/hft-ops"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FeatureCarrierAdmissionIssueError(RuntimeError):
    """hft-ops cannot establish every prerequisite for exact admission."""


@dataclass(frozen=True)
class IssuedFeatureCarrierAdmissionV1:
    path: Path
    admission_id: str
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
        raise FeatureCarrierAdmissionIssueError(
            f"git {' '.join(arguments)} failed in {repository}: "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise FeatureCarrierAdmissionIssueError(
            f"git {' '.join(arguments)} failed in {repository}: "
            f"{completed.stderr.decode(errors='replace').strip()}"
        )
    return completed.stdout


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


def _stable_file_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise FeatureCarrierAdmissionIssueError(
            f"identity input is absent, non-file, or symlink: {path}"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1024 * 1024):
                chunks.append(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        live = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise FeatureCarrierAdmissionIssueError(
            f"cannot hash stable identity input {path}: {exc}"
        ) from exc
    identities = {
        (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns)
        for row in (before, after, live)
    }
    if len(identities) != 1:
        raise FeatureCarrierAdmissionIssueError(
            f"identity input changed while hashing: {path}"
        )
    return b"".join(chunks)


def _stable_file_sha256(path: Path) -> str:
    return hashlib.sha256(_stable_file_bytes(path)).hexdigest()


def _dependency_graph_sha256(lock_payload: bytes, hft_ops_root: Path) -> str:
    try:
        lock = tomllib.loads(lock_payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise FeatureCarrierAdmissionIssueError(
            f"hft-ops dependency lock is malformed: {exc}"
        ) from exc
    packages = lock.get("package")
    if type(packages) is not list or not packages:
        raise FeatureCarrierAdmissionIssueError(
            "hft-ops dependency lock has no packages"
        )
    graph = []
    for index, package in enumerate(packages):
        if type(package) is not dict:
            raise FeatureCarrierAdmissionIssueError(
                f"hft-ops dependency lock package[{index}] is malformed"
            )
        name = package.get("name")
        version = package.get("version")
        source = package.get("source")
        if (
            type(name) is not str
            or type(version) is not str
            or type(source) is not dict
        ):
            raise FeatureCarrierAdmissionIssueError(
                f"hft-ops dependency lock package[{index}] lacks identity"
            )
        identity: dict[str, Any] = {
            "name": name,
            "version": version,
            "source": source,
        }
        directory = source.get("directory")
        if directory is not None:
            if type(directory) is not str:
                raise FeatureCarrierAdmissionIssueError(
                    f"hft-ops dependency lock package[{index}] directory is malformed"
                )
            try:
                dependency_root = (hft_ops_root / directory).resolve(strict=True)
            except OSError as exc:
                raise FeatureCarrierAdmissionIssueError(
                    f"locked local dependency is unavailable: {directory}: {exc}"
                ) from exc
            if _git(dependency_root, "status", "--porcelain", "--untracked-files=all"):
                raise FeatureCarrierAdmissionIssueError(
                    f"local dependency checkout is dirty: {dependency_root}"
                )
            local_repository = _canonical_github_repository(
                _git(dependency_root, "remote", "get-url", "origin")
            )
            local_commit = _git(dependency_root, "rev-parse", "HEAD")
            local_tree = _git(dependency_root, "rev-parse", "HEAD^{tree}")
            if (
                local_repository is None
                or re.fullmatch(r"[0-9a-f]{40}", local_commit) is None
                or re.fullmatch(r"[0-9a-f]{40}", local_tree) is None
            ):
                raise FeatureCarrierAdmissionIssueError(
                    f"locked local dependency Git identity is malformed: {dependency_root}"
                )
            identity["local_repository"] = local_repository
            identity["local_commit"] = local_commit
            identity["local_tree"] = local_tree
        graph.append(identity)
    graph.sort(
        key=lambda row: (row["name"], row["version"], portable_sha256(row["source"]))
    )
    return portable_sha256({"schema": "uv_dependency_graph_v1", "packages": graph})


def _tracked_hft_ops_source_sha256(hft_ops_root: Path) -> str:
    module_path = Path(__file__).resolve(strict=True)
    try:
        module_relative = module_path.relative_to(hft_ops_root).as_posix()
    except ValueError as exc:
        raise FeatureCarrierAdmissionIssueError(
            "executing admission issuer is outside the claimed hft-ops checkout"
        ) from exc
    tracked = [
        value.decode("utf-8")
        for value in _git_bytes(
            hft_ops_root, "ls-files", "-z", "--", "src/hft_ops"
        ).split(b"\0")
        if value
    ]
    if module_relative not in tracked:
        raise FeatureCarrierAdmissionIssueError(
            "executing admission issuer is not tracked by hft-ops HEAD"
        )
    files = []
    for relative in tracked:
        live = _stable_file_bytes(hft_ops_root / relative)
        committed = _git_bytes(hft_ops_root, "show", f"HEAD:{relative}")
        if live != committed:
            raise FeatureCarrierAdmissionIssueError(
                f"tracked hft-ops authority source differs from HEAD: {relative}"
            )
        files.append(
            {"relative_path": relative, "sha256": hashlib.sha256(live).hexdigest()}
        )
    return portable_sha256({"schema": "hft_ops_python_source_tree_v1", "files": files})


def _validate_runtime_contract_dependency(
    lock_payload: bytes, hft_ops_root: Path
) -> None:
    lock = tomllib.loads(lock_payload.decode("utf-8"))
    matches = [
        package
        for package in lock.get("package", [])
        if type(package) is dict and package.get("name") == "hft-contracts"
    ]
    if len(matches) != 1:
        raise FeatureCarrierAdmissionIssueError(
            "hft-ops lock must contain exactly one hft-contracts package"
        )
    source = matches[0].get("source")
    directory = source.get("directory") if type(source) is dict else None
    if type(directory) is not str:
        raise FeatureCarrierAdmissionIssueError(
            "hft-contracts must be an exact local locked dependency"
        )
    expected = (hft_ops_root / directory).resolve(strict=True)
    runtime_module = Path(
        validate_feature_carrier_admission_receipt.__code__.co_filename
    ).resolve(strict=True)
    try:
        runtime_module.relative_to(expected)
    except ValueError as exc:
        raise FeatureCarrierAdmissionIssueError(
            "runtime hft-contracts code differs from the locked local dependency"
        ) from exc


def _authority_identity(hft_ops_root: Path) -> dict[str, Any]:
    top_level = Path(_git(hft_ops_root, "rev-parse", "--show-toplevel")).resolve(
        strict=True
    )
    if top_level != hft_ops_root:
        raise FeatureCarrierAdmissionIssueError(
            "claimed hft-ops root is not the executing Git top level"
        )
    remote = _canonical_github_repository(
        _git(hft_ops_root, "remote", "get-url", "origin")
    )
    if remote != HFT_OPS_REPOSITORY:
        raise FeatureCarrierAdmissionIssueError(f"hft-ops origin mismatch: {remote!r}")
    status = _git(hft_ops_root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise FeatureCarrierAdmissionIssueError(
            "hft-ops admission authority checkout is dirty"
        )
    commit = _git(hft_ops_root, "rev-parse", "HEAD")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise FeatureCarrierAdmissionIssueError("hft-ops HEAD is not exact 40-hex")
    lock_path = hft_ops_root / "uv.lock"
    if _git(hft_ops_root, "ls-files", "--error-unmatch", "uv.lock") != "uv.lock":
        raise FeatureCarrierAdmissionIssueError("hft-ops uv.lock is not tracked")
    lock_payload = _stable_file_bytes(lock_path)
    if lock_payload != _git_bytes(hft_ops_root, "show", "HEAD:uv.lock"):
        raise FeatureCarrierAdmissionIssueError("hft-ops uv.lock differs from HEAD")
    lock_sha256 = hashlib.sha256(lock_payload).hexdigest()
    _validate_runtime_contract_dependency(lock_payload, hft_ops_root)
    return {
        "repository": HFT_OPS_REPOSITORY,
        "commit": commit,
        "git_dirty": False,
        "binary_sha256": _tracked_hft_ops_source_sha256(hft_ops_root),
        "dependency_lock_sha256": lock_sha256,
        "dependency_graph_sha256": _dependency_graph_sha256(lock_payload, hft_ops_root),
        "toolchain_id": (
            f"python{sys.version_info.major}_{sys.version_info.minor}_"
            f"{sys.version_info.micro}_{sys.implementation.name}"
        ),
    }


def _same_candidate(
    first: ResolvedFeatureCarrierAdmissionCandidateV1,
    second: ResolvedFeatureCarrierAdmissionCandidateV1,
) -> bool:
    return (
        first.generation_id == second.generation_id
        and first.generation_receipt_sha256 == second.generation_receipt_sha256
        and first.acceptance_receipt_sha256 == second.acceptance_receipt_sha256
        and first.generation_content == second.generation_content
        and first.acceptance_content == second.acceptance_content
        and first.acceptance_evidence == second.acceptance_evidence
        and first.acceptance_qualifications == second.acceptance_qualifications
        and first.build_trust_policy == second.build_trust_policy
        and first.build_trust_binding == second.build_trust_binding
        and first.acceptance_proven_gates == second.acceptance_proven_gates
    )


def issue_feature_carrier_admission(
    *,
    generation_root: Path,
    acceptance_receipt: str,
    hft_ops_root: Path,
    trusted_build_policy: TrustedFeatureCarrierBuildPolicyV1,
    admitted_at_utc: str,
) -> IssuedFeatureCarrierAdmissionV1:
    """Issue one immutable admission after two exact candidate resolutions."""

    if not FEATURE_CARRIER_PRODUCTION_ADMISSION_ENABLED:
        raise FeatureCarrierAdmissionIssueError(
            "schema-v4 feature-carrier production admission is disabled by the generated contract authority"
        )

    if generation_root.is_symlink() or hft_ops_root.is_symlink():
        raise FeatureCarrierAdmissionIssueError(
            "admission roots must be direct non-symlink directories"
        )
    try:
        generation_root = generation_root.resolve(strict=True)
        hft_ops_root = hft_ops_root.resolve(strict=True)
    except OSError as exc:
        raise FeatureCarrierAdmissionIssueError(
            f"admission root is unavailable: {exc}"
        ) from exc
    try:
        policy_path = trusted_build_policy.path.resolve(strict=True)
        policy_relative = policy_path.relative_to(hft_ops_root)
    except (OSError, ValueError) as exc:
        raise FeatureCarrierAdmissionIssueError(
            "trusted build policy is not inside the hft-ops authority root"
        ) from exc
    cursor = hft_ops_root
    for part in policy_relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise FeatureCarrierAdmissionIssueError(
                f"trusted build-policy path traverses a symlink: {cursor}"
            )

    try:
        first = resolve_feature_carrier_admission_candidate(
            generation_root,
            acceptance_receipt=acceptance_receipt,
            trusted_build_policy=trusted_build_policy,
        )
    except GenerationResolutionError as exc:
        raise FeatureCarrierAdmissionIssueError(
            f"admission candidate is invalid: {exc}"
        ) from exc
    expected_policy_relative = BUILD_TRUST_LEDGER_RELATIVE / (
        f"{first.build_trust_binding['policy_content_sha256']}.json"
    )
    if policy_relative != expected_policy_relative:
        raise FeatureCarrierAdmissionIssueError(
            "trusted build policy is not at its canonical hft-ops ledger path"
        )
    authority = _authority_identity(hft_ops_root)
    if first.build_trust_policy.get("authority") != authority:
        raise FeatureCarrierAdmissionIssueError(
            "running hft-ops authority differs from the externally trusted policy authority"
        )
    try:
        second = resolve_feature_carrier_admission_candidate(
            generation_root,
            acceptance_receipt=acceptance_receipt,
            trusted_build_policy=trusted_build_policy,
        )
    except GenerationResolutionError as exc:
        raise FeatureCarrierAdmissionIssueError(
            f"admission candidate became invalid: {exc}"
        ) from exc
    if not _same_candidate(first, second):
        raise FeatureCarrierAdmissionIssueError(
            "candidate changed between pre-admission resolutions"
        )
    authority_recheck = _authority_identity(hft_ops_root)
    if authority_recheck != authority:
        raise FeatureCarrierAdmissionIssueError(
            "hft-ops authority changed during admission evaluation"
        )
    try:
        final_candidate = resolve_feature_carrier_admission_candidate(
            generation_root,
            acceptance_receipt=acceptance_receipt,
            trusted_build_policy=trusted_build_policy,
        )
    except GenerationResolutionError as exc:
        raise FeatureCarrierAdmissionIssueError(
            f"admission candidate changed before publication: {exc}"
        ) from exc
    if not _same_candidate(second, final_candidate):
        raise FeatureCarrierAdmissionIssueError(
            "candidate changed before admission publication"
        )

    generation_reference = {
        "path": first.generation_receipt.relative_to(generation_root).as_posix(),
        "sha256": first.generation_receipt_sha256,
    }
    acceptance_reference = {
        "path": first.acceptance_receipt.relative_to(generation_root).as_posix(),
        "sha256": first.acceptance_receipt_sha256,
    }
    content = {
        "artifact": ADMISSION_SCHEMA,
        "status": "admitted",
        "canonicalization_id": PORTABLE_CANONICALIZATION_ID,
        "policy_id": ADMISSION_POLICY_ID,
        "admitted_at_utc": admitted_at_utc,
        "admission_id": "pending",
        "authority": authority,
        "generation_id": first.generation_id,
        "generation_content_sha256": first.acceptance_content[
            "generation_content_sha256"
        ],
        "generation_receipt": generation_reference,
        "acceptance_id": first.acceptance_content["acceptance_id"],
        "acceptance_content_sha256": portable_sha256(first.acceptance_content),
        "acceptance_receipt": acceptance_reference,
        "build_trust": dict(first.build_trust_binding),
    }
    content["admission_id"] = compute_feature_carrier_admission_id(content)
    envelope = build_portable_envelope(ADMISSION_SCHEMA, content)
    try:
        validate_feature_carrier_admission_receipt(envelope)
    except ValueError as exc:
        raise FeatureCarrierAdmissionIssueError(
            f"generated admission violates its contract: {exc}"
        ) from exc
    target_relative = Path(ADMISSION_DIRECTORY) / f"{content['admission_id']}.json"
    target = generation_root / target_relative
    try:
        result = immutable_write_json_beneath(
            generation_root, target_relative, envelope
        )
    except ImmutablePublicationError as exc:
        raise FeatureCarrierAdmissionIssueError(
            f"immutable admission publication failed: {exc}"
        ) from exc
    try:
        post_candidate = resolve_feature_carrier_admission_candidate(
            generation_root,
            acceptance_receipt=acceptance_receipt,
            trusted_build_policy=trusted_build_policy,
        )
    except GenerationResolutionError as exc:
        raise FeatureCarrierAdmissionIssueError(
            "candidate changed while admission was published; the immutable "
            "admission is non-activatable and must be quarantined: "
            f"{exc}"
        ) from exc
    if not _same_candidate(final_candidate, post_candidate):
        raise FeatureCarrierAdmissionIssueError(
            "candidate changed while admission was published; the immutable "
            "admission is non-activatable and must be quarantined"
        )
    if _authority_identity(hft_ops_root) != authority:
        raise FeatureCarrierAdmissionIssueError(
            "hft-ops authority changed while admission was published; the "
            "immutable admission is non-activatable and must be quarantined"
        )
    return IssuedFeatureCarrierAdmissionV1(
        path=target,
        admission_id=content["admission_id"],
        content_sha256=envelope["content_sha256"],
        file_sha256=result.sha256,
        write_result=result,
    )


__all__ = [
    "FeatureCarrierAdmissionIssueError",
    "IssuedFeatureCarrierAdmissionV1",
    "issue_feature_carrier_admission",
]
