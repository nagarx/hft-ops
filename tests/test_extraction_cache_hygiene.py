"""Scope-safe hygiene (2026-06-01): H5 + M4 in ``scheduler/extraction_cache.py``.

**H5** — ``prepare_cache_key_inputs`` narrowed its config-load ``except`` from the
overly-broad ``(OSError, Exception)`` (which is just ``except Exception`` — it
masked *programming* bugs as a silent "cache disabled") to the specific expected
I/O + decode failures ``(OSError, UnicodeDecodeError, TOMLDecodeError)``. A genuine
programming error (a ``None`` path, a downstream ``TypeError``/``RuntimeError``)
now PROPAGATES loudly instead of silently degrading to non-cached extraction.
``UnicodeDecodeError`` is included deliberately: ``Path.read_text()`` raises it
(a ``ValueError`` subclass, NOT an ``OSError``) on a non-UTF-8 config, which is an
expected bad-input anomaly — a bare ``(OSError, TOMLDecodeError)`` would convert
that graceful case into a hard crash.

**M4** — the ``gc_cache`` orphan-staging EXCLUSION guard matched ``-.tmp`` but real
staging dirs are ``<key>.tmp-<pid>`` (a ``.tmp-`` infix; see ``:286``/``:749``), so
the guard never matched its intended pattern. This is *behaviorally inert today*
(the 64-hex length/hex check on the next line excludes the same dirs) — a
guard-correctness fix, NOT a live disk-leak fix (there is no orphan-sweep; orphan
staging dirs are deliberately never deleted, since a ``.tmp-<pid>`` dir may belong
to a live concurrent populate). The tests below are characterization locks: an
orphan staging dir must never enter the eviction set.
"""

from __future__ import annotations

import os

import pytest

from hft_ops.scheduler.extraction_cache import gc_cache, prepare_cache_key_inputs


# ---------------------------------------------------------------------------
# H5 — prepare_cache_key_inputs error narrowing
# ---------------------------------------------------------------------------
def _other_kwargs(tmp_path):
    """Valid-type kwargs for the args NOT reached before the config-load line."""
    return dict(
        extractor_dir=tmp_path,
        reconstructor_dir=tmp_path,
        hft_statistics_dir=None,
        contract_version="3.0",
        data_dir=tmp_path,
    )


def test_malformed_toml_disables_cache_gracefully(tmp_path):
    """A TOMLDecodeError is an EXPECTED anomaly -> return None (cache disabled)."""
    cfg = tmp_path / "bad.toml"
    cfg.write_text("this is = = not [valid toml")
    assert prepare_cache_key_inputs(cfg, **_other_kwargs(tmp_path)) is None


def test_non_utf8_config_disables_cache_gracefully(tmp_path):
    """read_text() on non-UTF-8 bytes raises UnicodeDecodeError (a ValueError, NOT
    an OSError) -> must still be handled gracefully. Locks the UnicodeDecodeError
    arm: a bare ``(OSError, TOMLDecodeError)`` would propagate (crash) here."""
    cfg = tmp_path / "binary.toml"
    cfg.write_bytes(b"\xff\xfe\x00\x01 not valid utf-8 \xc3\x28")
    assert prepare_cache_key_inputs(cfg, **_other_kwargs(tmp_path)) is None


def test_missing_config_disables_cache_gracefully(tmp_path):
    """FileNotFoundError (an OSError) -> return None (graceful)."""
    cfg = tmp_path / "does_not_exist.toml"
    assert prepare_cache_key_inputs(cfg, **_other_kwargs(tmp_path)) is None


def test_programming_error_propagates_not_silently_swallowed(tmp_path):
    """H5 bite: a programming error (``None`` path -> AttributeError on
    ``.read_text()``) MUST propagate, not be masked as a silent cache-disable.
    Under the old ``except (OSError, Exception)`` this returned ``None``."""
    with pytest.raises(AttributeError):
        prepare_cache_key_inputs(None, **_other_kwargs(tmp_path))


def test_parse_runtime_error_propagates(tmp_path, monkeypatch):
    """H5 bite #2: a non-OSError/non-decode error raised DURING parse (a
    downstream bug) propagates rather than silently disabling the cache."""
    cfg = tmp_path / "ok.toml"
    cfg.write_text("x = 1\n")
    try:
        import tomllib as _toml
    except ImportError:  # pragma: no cover — py<3.11 fallback path
        import tomli as _toml  # type: ignore

    def _boom(_s):
        raise RuntimeError("simulated downstream parse bug")

    monkeypatch.setattr(_toml, "loads", _boom)
    with pytest.raises(RuntimeError):
        prepare_cache_key_inputs(cfg, **_other_kwargs(tmp_path))


# ---------------------------------------------------------------------------
# M4 — gc_cache orphan-staging exclusion guard
# ---------------------------------------------------------------------------
_HEX64_A = "a" * 64
_HEX64_B = "b" * 64


def test_orphan_only_cache_yields_no_evictions(tmp_path):
    """An orphan ``<key>.tmp-<pid>`` staging dir is excluded from the candidate
    set, so a cache holding only an orphan evicts nothing (regardless of policy
    thresholds — the exclusion happens before any age/size filter)."""
    cache_root = tmp_path / "_cache"
    cache_root.mkdir()
    (cache_root / f"{_HEX64_A}.tmp-99999").mkdir()
    assert gc_cache(cache_root, older_than_days=0, dry_run=True) == []


def test_orphan_excluded_even_when_real_entry_evicted(tmp_path):
    """When a finalized entry IS eligible for eviction, the co-resident orphan
    staging dir must still be excluded from the eviction set."""
    cache_root = tmp_path / "_cache"
    cache_root.mkdir()
    finalized = cache_root / _HEX64_A
    finalized.mkdir()
    # Force the finalized entry clearly into the past so older_than_days bites.
    old = 1.0
    os.utime(finalized, (old, old))
    orphan = cache_root / f"{_HEX64_B}.tmp-12345"
    orphan.mkdir()

    evicted = {p.name for p in gc_cache(cache_root, older_than_days=1, dry_run=True)}
    assert _HEX64_A in evicted, "the aged finalized entry should be evictable"
    assert f"{_HEX64_B}.tmp-12345" not in evicted, (
        "orphan staging dirs must never be evicted as finalized entries"
    )
