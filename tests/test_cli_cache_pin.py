"""#PY-187 closure (2026-05-13) — regression test for cache pin
corruption-wipe at ``hft_ops.cli._update_pinned``.

Pre-fix: a corrupt ``_PINNED.json`` was silently swallowed via
``except (OSError, json.JSONDecodeError): pass`` → ``pinned`` defaulted to
``set()`` → the unconditional ``atomic_write_json`` that followed
OVERWROTE the corrupt file with an empty set. All operator-pinned cache
keys were silently wiped. No log, no error, no recovery path.

Post-fix: rotate the corrupt file to ``_PINNED.json.corrupt.<UTC-timestamp>``
and raise ``click.ClickException``. Operator workflow: inspect the
rotated file, decide to delete (to start fresh) or manually re-pin needed
keys, then re-run.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import pytest

from hft_ops.cli import _update_pinned


class TestUpdatePinnedHappyPath:
    """Baseline — pre-fix behavior for VALID inputs must be preserved."""

    def test_creates_new_pin_file_when_missing(self, tmp_path: Path) -> None:
        cache_root = tmp_path / "_cache"
        # No _PINNED.json exists yet.
        _update_pinned(cache_root, add="key_abc123")

        pin_file = cache_root / "_PINNED.json"
        assert pin_file.exists()
        data = json.loads(pin_file.read_text())
        assert data == {"pinned_keys": ["key_abc123"]}

    def test_adds_to_existing_valid_pin_file(self, tmp_path: Path) -> None:
        cache_root = tmp_path / "_cache"
        cache_root.mkdir()
        pin_file = cache_root / "_PINNED.json"
        pin_file.write_text(json.dumps({"pinned_keys": ["existing_key"]}))

        _update_pinned(cache_root, add="new_key")

        data = json.loads(pin_file.read_text())
        assert sorted(data["pinned_keys"]) == ["existing_key", "new_key"]

    def test_removes_from_existing_valid_pin_file(self, tmp_path: Path) -> None:
        cache_root = tmp_path / "_cache"
        cache_root.mkdir()
        pin_file = cache_root / "_PINNED.json"
        pin_file.write_text(json.dumps({"pinned_keys": ["a", "b", "c"]}))

        _update_pinned(cache_root, remove="b")

        data = json.loads(pin_file.read_text())
        assert sorted(data["pinned_keys"]) == ["a", "c"]


class TestUpdatePinnedCorruptionHandling:
    """#PY-187 regression — corrupt _PINNED.json must NOT silently wipe."""

    def test_corrupt_json_raises_click_exception(self, tmp_path: Path) -> None:
        """Operator runs `cache pin <key>` against corrupt _PINNED.json →
        must raise (not silently overwrite)."""
        cache_root = tmp_path / "_cache"
        cache_root.mkdir()
        pin_file = cache_root / "_PINNED.json"
        # Write invalid JSON
        pin_file.write_text("{not valid json[")

        with pytest.raises(click.ClickException) as exc_info:
            _update_pinned(cache_root, add="new_key")

        # Error message must be actionable
        msg = exc_info.value.message
        assert "corrupt" in msg.lower()
        assert "_PINNED.json" in msg

    def test_corrupt_json_rotates_to_backup(self, tmp_path: Path) -> None:
        """Corrupt file is rotated to _PINNED.json.corrupt.<timestamp>
        BEFORE the raise — so operator can inspect/recover."""
        cache_root = tmp_path / "_cache"
        cache_root.mkdir()
        pin_file = cache_root / "_PINNED.json"
        corrupt_content = "{this is not json"
        pin_file.write_text(corrupt_content)

        with pytest.raises(click.ClickException):
            _update_pinned(cache_root, add="new_key")

        # Original file should be rotated, not silently overwritten.
        rotated = list(cache_root.glob("_PINNED.json.corrupt.*"))
        assert len(rotated) == 1, (
            f"Expected exactly 1 rotated file, got {len(rotated)}: "
            f"{[p.name for p in rotated]}"
        )
        # Rotated file preserves the corrupt content.
        assert rotated[0].read_text() == corrupt_content
        # Original path was renamed (not still there with corrupt data).
        assert not pin_file.exists() or pin_file.read_text() != corrupt_content

    def test_corrupt_json_no_silent_overwrite(self, tmp_path: Path) -> None:
        """Locks the regression — after the raise, no empty
        _PINNED.json must exist at the original path (which would be
        the silent-wipe failure mode)."""
        cache_root = tmp_path / "_cache"
        cache_root.mkdir()
        pin_file = cache_root / "_PINNED.json"
        pin_file.write_text("not_json")

        with pytest.raises(click.ClickException):
            _update_pinned(cache_root, add="some_key")

        # If the file is back at the original path, it must NOT be the
        # silently-rewritten empty set (pre-fix failure mode).
        if pin_file.exists():
            content = pin_file.read_text()
            assert content != '{"pinned_keys": []}', (
                "#PY-187 regression: corrupt _PINNED.json was silently "
                "overwritten with empty pinned_keys set"
            )

    def test_subsequent_call_works_against_clean_state(self, tmp_path: Path) -> None:
        """After operator deletes rotated corrupt file, pin command
        succeeds against clean state."""
        cache_root = tmp_path / "_cache"
        cache_root.mkdir()
        pin_file = cache_root / "_PINNED.json"
        pin_file.write_text("corrupt")

        # First call: raises
        with pytest.raises(click.ClickException):
            _update_pinned(cache_root, add="key1")

        # Operator deletes rotated file (or doesn't — original is rotated
        # away regardless). Subsequent call against clean state should work.
        for rotated in cache_root.glob("_PINNED.json.corrupt.*"):
            rotated.unlink()

        # Now pin should succeed (file does not exist at original path).
        _update_pinned(cache_root, add="key1")
        assert pin_file.exists()
        data = json.loads(pin_file.read_text())
        assert data["pinned_keys"] == ["key1"]
