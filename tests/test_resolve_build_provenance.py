"""P1a (2026-05-30, finding A-PROV) — producer-provenance resolver tests.

``resolve_build_provenance`` is the FAIL-OPEN producer-code provenance resolver:
it captures the extractor / reconstructor / hft-statistics git shas (+ a
reconstructor source/dirty tag + a completeness marker) into the flat
``Dict[str, str]`` consumed by
``hft_contracts.provenance.Provenance.producer_commits``. It NEVER raises
(provenance is a record-level OBSERVATION, not a gate) and reuses the shared
``resolve_patched_crate_dir`` ``.cargo/config.toml`` parser (hft-rules §0), whose
git-URL-keyed parse ALSO fixes the previously-dead ``_resolve_hft_statistics_sha``
tier-2 (which read the wrong ``patch.crates-io.<crate>.path`` key).

Tests use isolated tmp git repos so they are deterministic and never touch the
real pipeline checkouts.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from hft_ops.scheduler.extraction_cache import (
    _resolve_hft_statistics_sha,
    resolve_build_provenance,
    resolve_patched_crate_dir,
)

# The 5 keys locked by ``hft_contracts.provenance.Provenance.producer_commits``
# (docstring + ``test_provenance.py::TestProvenanceProducerCommits``). The
# resolver MUST emit exactly this vocabulary so the two repos cannot drift.
_EXPECTED_KEYS = {
    "extractor_git_sha",
    "reconstructor_git_sha",
    "hft_statistics_git_sha",
    "reconstructor_source",
    "completeness",
}

_UNRESOLVED = "unresolved"


# ---------------------------------------------------------------------------
# Helpers (deterministic tmp git repos)
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )


def _init_git_repo(repo: Path) -> str:
    """Create a minimal committed git repo at ``repo``; return its HEAD sha."""
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.t")
    _git(repo, "config", "user.name", "t")
    (repo / "f.txt").write_text("x")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-q", "-m", "init")
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo, check=True, capture_output=True, text=True,
    )
    return out.stdout.strip()


def _write_cargo_config(
    extractor_dir: Path, recon_path: str, hftstats_path: str
) -> None:
    """Write a realistic git-URL-keyed ``.cargo/config.toml`` (matches the real
    ``feature-extractor-MBO-LOB/.cargo/config.toml`` table shape)."""
    cargo = extractor_dir / ".cargo"
    cargo.mkdir(parents=True, exist_ok=True)
    (cargo / "config.toml").write_text(
        '[patch."https://github.com/nagarx/MBO-LOB-reconstructor.git"]\n'
        f'mbo-lob-reconstructor = {{ path = "{recon_path}" }}\n\n'
        '[patch."https://github.com/nagarx/hft-statistics.git"]\n'
        f'hft-statistics = {{ path = "{hftstats_path}" }}\n'
    )


# ---------------------------------------------------------------------------
# resolve_build_provenance — vocabulary + fail-open
# ---------------------------------------------------------------------------


def test_returns_locked_key_vocabulary(tmp_path: Path):
    """Always returns exactly the 5 keys locked by the Provenance contract."""
    out = resolve_build_provenance(
        extractor_dir=tmp_path / "ext",
        reconstructor_dir=tmp_path / "recon",
    )
    assert set(out.keys()) == _EXPECTED_KEYS


def test_partial_when_dirs_missing(tmp_path: Path):
    """FAIL-OPEN: non-git dirs → every sha ``unresolved``, completeness
    ``partial``, source ``git-pin`` (no .cargo override). Never raises."""
    out = resolve_build_provenance(
        extractor_dir=tmp_path / "ext",
        reconstructor_dir=tmp_path / "recon",
    )
    assert out["extractor_git_sha"] == _UNRESOLVED
    assert out["reconstructor_git_sha"] == _UNRESOLVED
    assert out["hft_statistics_git_sha"] == _UNRESOLVED
    assert out["completeness"] == "partial"
    assert out["reconstructor_source"] == "git-pin"


def test_never_raises_on_garbage_paths():
    """Provenance is an observation, not a gate — garbage in, dict out."""
    out = resolve_build_provenance(
        extractor_dir=Path("/nonexistent/zzz"),
        reconstructor_dir=Path("/nonexistent/yyy"),
    )
    assert set(out.keys()) == _EXPECTED_KEYS
    assert out["completeness"] == "partial"


def test_full_capture_clean_with_path_override(tmp_path: Path):
    """All three shas resolve via git-URL path overrides → completeness ``full``;
    a clean reconstructor working tree → ``path-override@<sha>+clean``."""
    ext = tmp_path / "feature-extractor-MBO-LOB"
    recon = tmp_path / "MBO-LOB-reconstructor"
    hftstats = tmp_path / "hft-statistics"
    ext_sha = _init_git_repo(ext)
    recon_sha = _init_git_repo(recon)
    hftstats_sha = _init_git_repo(hftstats)
    # .cargo/config.toml lives in ext; override paths are relative to ext.
    _write_cargo_config(
        ext, recon_path="../MBO-LOB-reconstructor", hftstats_path="../hft-statistics"
    )

    out = resolve_build_provenance(extractor_dir=ext, reconstructor_dir=recon)

    assert out["extractor_git_sha"] == ext_sha
    assert out["reconstructor_git_sha"] == recon_sha
    assert out["hft_statistics_git_sha"] == hftstats_sha
    assert out["completeness"] == "full"
    assert out["reconstructor_source"] == f"path-override@{recon_sha}+clean"


def test_dirty_reconstructor_tagged_dirty(tmp_path: Path):
    """An uncommitted change in the reconstructor checkout → ``+dirty`` tag (the
    real-pipeline state: the override target has untracked working-tree files)."""
    ext = tmp_path / "feature-extractor-MBO-LOB"
    recon = tmp_path / "MBO-LOB-reconstructor"
    _init_git_repo(ext)
    recon_sha = _init_git_repo(recon)
    (recon / "untracked.txt").write_text("dirty")  # working tree now dirty
    _write_cargo_config(ext, "../MBO-LOB-reconstructor", "../hft-statistics")

    out = resolve_build_provenance(extractor_dir=ext, reconstructor_dir=recon)
    assert out["reconstructor_source"] == f"path-override@{recon_sha}+dirty"


def test_git_pin_when_no_cargo_override(tmp_path: Path):
    """No ``.cargo/config.toml`` → ``reconstructor_source`` is ``git-pin`` (the
    CI build mechanism). In git-pin mode the build uses the Cargo.toml pin, NOT
    the local checkout, so the local HEAD is deliberately NOT reported as
    ``reconstructor_git_sha`` (reporting it would be silently-wrong lineage —
    matches the Provenance.producer_commits docstring). The sha is ``unresolved``
    and ``completeness`` is therefore ``partial`` even though the extractor
    resolved."""
    ext = tmp_path / "feature-extractor-MBO-LOB"
    recon = tmp_path / "MBO-LOB-reconstructor"
    _init_git_repo(ext)
    _init_git_repo(recon)  # checkout exists but is NOT the build source in git-pin
    # Intentionally NO .cargo/config.toml.

    out = resolve_build_provenance(extractor_dir=ext, reconstructor_dir=recon)
    assert out["reconstructor_source"] == "git-pin"
    assert out["reconstructor_git_sha"] == _UNRESOLVED
    assert out["completeness"] == "partial"


def test_extractor_unresolved_marks_partial(tmp_path: Path):
    """Any single unresolved producer → completeness ``partial`` (a non-git
    extractor with everything else resolvable)."""
    ext = tmp_path / "feature-extractor-MBO-LOB"  # NOT a git repo
    ext.mkdir()
    recon = tmp_path / "MBO-LOB-reconstructor"
    hftstats = tmp_path / "hft-statistics"
    recon_sha = _init_git_repo(recon)
    _init_git_repo(hftstats)
    _write_cargo_config(ext, "../MBO-LOB-reconstructor", "../hft-statistics")

    out = resolve_build_provenance(extractor_dir=ext, reconstructor_dir=recon)
    assert out["extractor_git_sha"] == _UNRESOLVED
    assert out["reconstructor_git_sha"] == recon_sha
    assert out["completeness"] == "partial"


def test_path_override_with_unresolvable_reconstructor(tmp_path: Path):
    """Override configured (``.cargo`` patch present) but the reconstructor
    checkout is NOT a git repo → sha ``unresolved``, dirty ``unknown`` → source
    ``path-override@unresolved+unknown``, completeness ``partial``. Locks the
    messiest reachable source-tag form AND the ``unknown`` dirty branch."""
    ext = tmp_path / "feature-extractor-MBO-LOB"
    ext.mkdir()
    recon = tmp_path / "MBO-LOB-reconstructor"
    recon.mkdir()  # exists so the override path resolves, but NOT a git repo
    _write_cargo_config(ext, "../MBO-LOB-reconstructor", "../hft-statistics")

    out = resolve_build_provenance(extractor_dir=ext, reconstructor_dir=recon)
    assert out["reconstructor_git_sha"] == _UNRESOLVED
    assert out["reconstructor_source"] == "path-override@unresolved+unknown"
    assert out["completeness"] == "partial"


# ---------------------------------------------------------------------------
# resolve_patched_crate_dir — the SSoT .cargo parser (git-URL key bug-fix)
# ---------------------------------------------------------------------------


def test_resolve_patched_crate_dir_git_url_key(tmp_path: Path):
    """The SSoT parser resolves git-URL-keyed ``[patch."https://…"]`` tables —
    the case the pre-fix ``crates-io`` parser silently missed."""
    ext = tmp_path / "ext"
    ext.mkdir()
    _write_cargo_config(ext, "../MBO-LOB-reconstructor", "../hft-statistics")

    got_hft = resolve_patched_crate_dir(ext, "hft-statistics", "hft-statistics.git")
    assert got_hft == (ext / "../hft-statistics").resolve()

    got_recon = resolve_patched_crate_dir(
        ext, "mbo-lob-reconstructor", "MBO-LOB-reconstructor.git"
    )
    assert got_recon == (ext / "../MBO-LOB-reconstructor").resolve()


def test_resolve_patched_crate_dir_missing_config(tmp_path: Path):
    """No ``.cargo/config.toml`` → None (fail-soft)."""
    assert (
        resolve_patched_crate_dir(
            tmp_path / "ext", "hft-statistics", "hft-statistics.git"
        )
        is None
    )


def test_resolve_patched_crate_dir_ignores_crates_io_table(tmp_path: Path):
    """Regression for the fixed bug: the parser keys off the git-URL substring,
    so a legacy ``[patch.crates-io]`` table (which the OLD code wrongly read) is
    correctly NOT matched — it has no git-URL key."""
    ext = tmp_path / "ext"
    (ext / ".cargo").mkdir(parents=True)
    (ext / ".cargo" / "config.toml").write_text(
        '[patch.crates-io]\nhft-statistics = { path = "../hft-statistics" }\n'
    )
    assert (
        resolve_patched_crate_dir(ext, "hft-statistics", "hft-statistics.git")
        is None
    )


def test_resolve_patched_crate_dir_corrupt_toml(tmp_path: Path):
    """Corrupt TOML → None (fail-soft, never raises)."""
    ext = tmp_path / "ext"
    (ext / ".cargo").mkdir(parents=True)
    (ext / ".cargo" / "config.toml").write_text("this is = = not valid toml [[[")
    assert (
        resolve_patched_crate_dir(ext, "hft-statistics", "hft-statistics.git")
        is None
    )


def test_resolve_patched_crate_dir_non_string_path_is_failsoft(tmp_path: Path):
    """A syntactically-valid TOML with a truthy NON-string ``path`` (e.g. an int)
    must NOT crash ``PosixPath / <non-str>`` — the parser returns None and the
    fail-open resolver still returns a dict (regression for the isinstance(str)
    guard; the 'NEVER raises' contract)."""
    ext = tmp_path / "feature-extractor-MBO-LOB"
    (ext / ".cargo").mkdir(parents=True)
    (ext / ".cargo" / "config.toml").write_text(
        '[patch."https://github.com/nagarx/hft-statistics.git"]\n'
        "hft-statistics = { path = 12345 }\n"  # int, not str — valid TOML
    )
    # Parser is fail-soft (no TypeError):
    assert (
        resolve_patched_crate_dir(ext, "hft-statistics", "hft-statistics.git")
        is None
    )
    # And the top-level resolver never raises on such a config:
    out = resolve_build_provenance(
        extractor_dir=ext, reconstructor_dir=tmp_path / "recon"
    )
    assert set(out.keys()) == _EXPECTED_KEYS


# ---------------------------------------------------------------------------
# _resolve_hft_statistics_sha — git-URL tier-2 bug-fix regression
# ---------------------------------------------------------------------------


def test_resolve_hft_statistics_sha_via_git_url_override(tmp_path: Path):
    """Prove the FIXED tier-2 (git-URL key) resolves INDEPENDENTLY of the tier-3
    sibling fallback: put the override at a NON-sibling path so only tier-2 can
    find it. The pre-fix ``crates-io`` parser returned None here → tier-3 sibling
    miss → None (cache silently disabled). Now it resolves the true sha.
    """
    ext = tmp_path / "feature-extractor-MBO-LOB"
    ext.mkdir(parents=True)
    custom = tmp_path / "hftstats-custom"  # NOT the sibling location
    sha = _init_git_repo(custom)
    (ext / ".cargo").mkdir()
    (ext / ".cargo" / "config.toml").write_text(
        '[patch."https://github.com/nagarx/hft-statistics.git"]\n'
        'hft-statistics = { path = "../hftstats-custom" }\n'
    )
    # The tier-3 sibling (ext.parent / "hft-statistics") does NOT exist, so only
    # the (fixed) tier-2 git-URL override can resolve the sha.
    assert not (ext.parent / "hft-statistics").exists()

    assert _resolve_hft_statistics_sha(None, ext) == sha


def test_resolve_hft_statistics_sha_explicit_dir_wins(tmp_path: Path):
    """Tier-1 explicit dir short-circuits the .cargo parse (unchanged behavior)."""
    explicit = tmp_path / "explicit-hftstats"
    sha = _init_git_repo(explicit)
    ext = tmp_path / "ext"
    ext.mkdir()
    assert _resolve_hft_statistics_sha(explicit, ext) == sha


# ---------------------------------------------------------------------------
# Wiring: extraction captured_metrics → ExperimentRecord.provenance (cli)
# ---------------------------------------------------------------------------


def test_producer_commits_flows_into_record(tmp_path: Path):
    """End-to-end: a ``producer_commits`` dict captured by the extraction stage
    in ``captured_metrics`` is harvested by ``cli._record_experiment`` and stored
    on ``ExperimentRecord.provenance.producer_commits`` (the A-PROV plumbing)."""
    from hft_ops.cli import _record_experiment
    from hft_ops.ledger import ExperimentLedger
    from hft_ops.manifest.schema import ExperimentHeader, ExperimentManifest
    from hft_ops.paths import PipelinePaths
    from hft_ops.stages.base import StageResult, StageStatus

    paths = PipelinePaths(pipeline_root=tmp_path)

    commits = {
        "extractor_git_sha": "e43bff0",
        "reconstructor_git_sha": "2b74523",
        "hft_statistics_git_sha": "db48275",
        "reconstructor_source": "path-override@2b74523+dirty",
        "completeness": "full",
    }
    extraction_result = StageResult(stage_name="extraction")
    extraction_result.status = StageStatus.COMPLETED
    extraction_result.captured_metrics = {"producer_commits": commits}

    results = {"extraction": extraction_result}
    manifest = ExperimentManifest(
        experiment=ExperimentHeader(name="test_producer_commits_wiring"),
    )

    experiment_id = _record_experiment(
        manifest, paths, fingerprint="d" * 64,
        results=results, total_duration=1.0,
    )

    record = ExperimentLedger(paths.ledger_dir).get(experiment_id)
    assert record is not None
    assert record.provenance.producer_commits == commits


def test_producer_commits_empty_when_extraction_absent(tmp_path: Path):
    """No extraction stage (e.g. training-only run) → ``producer_commits`` is
    ``{}`` — honest not-applicable; the field defaults empty (back-compat)."""
    from hft_ops.cli import _record_experiment
    from hft_ops.ledger import ExperimentLedger
    from hft_ops.manifest.schema import ExperimentHeader, ExperimentManifest
    from hft_ops.paths import PipelinePaths
    from hft_ops.stages.base import StageResult, StageStatus

    paths = PipelinePaths(pipeline_root=tmp_path)
    training_result = StageResult(stage_name="training")
    training_result.status = StageStatus.COMPLETED
    training_result.captured_metrics = {"test_ic": 0.38}

    results = {"training": training_result}
    manifest = ExperimentManifest(
        experiment=ExperimentHeader(name="test_producer_commits_absent"),
    )
    experiment_id = _record_experiment(
        manifest, paths, fingerprint="e" * 64,
        results=results, total_duration=1.0,
    )

    record = ExperimentLedger(paths.ledger_dir).get(experiment_id)
    assert record is not None
    assert record.provenance.producer_commits == {}


def test_extraction_stage_run_populates_producer_commits(tmp_path, monkeypatch):
    """Component/integration: ``ExtractionRunner.run()`` on the subprocess-
    COMPLETED path actually writes ``captured_metrics["producer_commits"]`` with
    the locked 5-key vocabulary. The other wiring tests stop at a hand-built
    ``StageResult`` + ``cli._record_experiment`` — THIS exercises the real stage
    capture site (extraction.py; the cache-HIT site is the identical call), so a
    refactor that drops the capture line MUST fail here. The fake subprocess
    never invokes cargo; the tmp pipeline_root has no git checkouts, so the
    resolver fails OPEN (``partial``) rather than crashing the completed stage —
    locking that the capture's argument evaluation (config.paths.* property
    joins) is safe in the real run path."""
    import types

    from hft_ops.config import OpsConfig
    from hft_ops.manifest.schema import (
        ExperimentHeader,
        ExperimentManifest,
        ExtractionStage,
        Stages,
    )
    from hft_ops.stages.base import StageStatus
    from hft_ops.stages.extraction import ExtractionRunner

    # Fake a successful cargo build (rc=0); never actually shells out.
    monkeypatch.setattr(
        "hft_ops.stages.extraction.run_subprocess",
        lambda cmd, cwd=None, verbose=False, env=None: types.SimpleNamespace(
            returncode=0, stdout="ok", stderr=""
        ),
    )

    config = OpsConfig.from_pipeline_root(
        pipeline_root=tmp_path, cache_extraction=False  # skip the cache-consult path
    )
    manifest = ExperimentManifest(
        experiment=ExperimentHeader(name="test_extraction_producer_commits"),
        stages=Stages(
            extraction=ExtractionStage(
                config="feature-extractor-MBO-LOB/configs/x.toml",
                output_dir="data/exports/test_out",  # does NOT exist → no skip
            ),
        ),
    )

    result = ExtractionRunner().run(manifest, config)

    assert result.status == StageStatus.COMPLETED
    pc = result.captured_metrics.get("producer_commits")
    assert isinstance(pc, dict), "extraction stage must populate producer_commits"
    assert set(pc.keys()) == _EXPECTED_KEYS  # locked vocabulary — no drift
    # tmp pipeline_root has no git checkouts → fail-open partial, NOT a crash:
    assert pc["completeness"] == "partial"


def test_extraction_cache_hit_populates_producer_commits(tmp_path, monkeypatch):
    """Component/integration: ``ExtractionRunner.run()`` on the cache-HIT path
    (extraction.py:141) ALSO writes ``captured_metrics["producer_commits"]`` with
    the locked 5-key vocabulary.

    ``test_extraction_stage_run_populates_producer_commits`` locks the
    completed-subprocess capture (extraction.py:200); THIS locks the SECOND,
    structurally-distinct capture site — a refactor that drops ONLY the cache-hit
    capture line would leave cache-hit experiment records with empty
    ``producer_commits`` yet pass every other test. The three cache calls are
    stubbed to force a deterministic hit without the real cache machinery; the
    subprocess is wired to explode if reached (the hit path must NOT shell out)."""
    import types

    from hft_ops.config import OpsConfig
    from hft_ops.manifest.schema import (
        ExperimentHeader,
        ExperimentManifest,
        ExtractionStage,
        Stages,
    )
    from hft_ops.stages.base import StageStatus
    from hft_ops.stages.extraction import ExtractionRunner

    # Force a deterministic cache HIT without the real cache machinery.
    monkeypatch.setattr(
        "hft_ops.stages.extraction.prepare_cache_key_inputs",
        lambda **kw: types.SimpleNamespace(),  # opaque; never inspected on the hit path
    )
    monkeypatch.setattr(
        "hft_ops.stages.extraction.compute_cache_key",
        lambda inputs: "a" * 64,
    )
    monkeypatch.setattr(
        "hft_ops.stages.extraction.resolve_or_link",
        lambda key, out, root: types.SimpleNamespace(
            status="hit", seconds_saved=1.0, linked_files=8, link_type="reflink"
        ),
    )
    # On a cache hit the stage must NOT shell out — blow up if it tries.
    monkeypatch.setattr(
        "hft_ops.stages.extraction.run_subprocess",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("subprocess ran on a cache hit")
        ),
    )

    config = OpsConfig.from_pipeline_root(pipeline_root=tmp_path, cache_extraction=True)
    manifest = ExperimentManifest(
        experiment=ExperimentHeader(name="test_extraction_cache_hit_pc"),
        stages=Stages(
            extraction=ExtractionStage(
                config="feature-extractor-MBO-LOB/configs/x.toml",
                output_dir="data/exports/test_out_hit",  # not None → cache path eligible
            ),
        ),
    )

    result = ExtractionRunner().run(manifest, config)

    assert result.status == StageStatus.SKIPPED  # cache hit → SKIPPED
    assert result.captured_metrics.get("cache_hit") is True  # proves the hit branch ran
    pc = result.captured_metrics.get("producer_commits")
    assert isinstance(pc, dict), "cache-hit path must populate producer_commits"
    assert set(pc.keys()) == _EXPECTED_KEYS  # locked vocabulary — no drift
    # tmp pipeline_root has no git checkouts → fail-open partial, NOT a crash:
    assert pc["completeness"] == "partial"
