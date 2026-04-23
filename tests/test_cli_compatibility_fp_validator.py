"""Phase V.1.5 follow-up (2026-04-23): SDR-3 fail-loud CLI validator tests.

Locks the contract on ``--compatibility-fp`` option (and any future
command reusing ``_validate_content_hash_option``):

1. Option absent (value=None) → pass through unchanged.
2. Valid 64-lowercase-hex → pass through unchanged.
3. Malformed input (uppercase, short, non-hex, empty string, prefix) →
   ``click.BadParameter`` with message citing the offending value.

Before this fix, ``hft-ops ledger list --compatibility-fp ABC123...`` would
exact-match against stored lowercase fingerprints, silently return zero
records, and give the operator NO signal that the input was malformed —
a hft-rules §5 silent-degrade. Fail-loud is now CLI-level.

These are COMPONENT + E2E tests combined — component (direct callback
invocation) covers the validator logic; E2E (via CliRunner invoking
``hft-ops ledger list --compatibility-fp BAD``) covers the Click wiring.
"""

from __future__ import annotations

import pytest
import click
from click.testing import CliRunner

from hft_ops.cli import _validate_content_hash_option, main


# =============================================================================
# COMPONENT: direct callback invocation (no Click machinery)
# =============================================================================


class _FakeParam:
    """Minimal stand-in for click.Parameter used only for BadParameter ctor."""
    name = "compatibility_fp"
    human_readable_name = "--compatibility-fp"
    opts = ["--compatibility-fp"]


class TestValidateContentHashOptionCallback:
    """Component-level tests of the Click callback function."""

    def test_none_passes_through(self):
        """Option not supplied → return None unchanged."""
        result = _validate_content_hash_option(None, _FakeParam(), None)
        assert result is None

    def test_valid_lowercase_hex_passes_through(self):
        """64 lowercase hex chars → return value unchanged (no normalization)."""
        valid = "a" * 64
        result = _validate_content_hash_option(None, _FakeParam(), valid)
        assert result == valid

    def test_valid_mixed_lowercase_hex_passes_through(self):
        """Realistic SHA-256 with mixed 0-9a-f characters."""
        valid = "96f60276abcdef0123456789abcdef0123456789abcdef0123456789abcdfc28"
        assert len(valid) == 64
        result = _validate_content_hash_option(None, _FakeParam(), valid)
        assert result == valid

    def test_uppercase_rejected(self):
        """SHA-256 is conventionally lowercase; uppercase must fail-loud."""
        with pytest.raises(click.BadParameter) as exc_info:
            _validate_content_hash_option(None, _FakeParam(), "A" * 64)
        assert "64 lowercase hex" in str(exc_info.value.message)
        # Value is surfaced in the error so operator can see what they pasted
        assert "'AAAAAAAAA" in str(exc_info.value.message)

    def test_truncated_hex_rejected(self):
        """63 chars (one too few) must fail-loud."""
        with pytest.raises(click.BadParameter):
            _validate_content_hash_option(None, _FakeParam(), "a" * 63)

    def test_overlong_hex_rejected(self):
        """65 chars (one too many) must fail-loud."""
        with pytest.raises(click.BadParameter):
            _validate_content_hash_option(None, _FakeParam(), "a" * 65)

    def test_non_hex_chars_rejected(self):
        """Values with g-z characters (outside 0-9a-f) must fail-loud."""
        with pytest.raises(click.BadParameter):
            _validate_content_hash_option(None, _FakeParam(), "z" * 64)

    def test_empty_string_rejected(self):
        """Empty string is not None — treat as malformed explicit input."""
        with pytest.raises(click.BadParameter):
            _validate_content_hash_option(None, _FakeParam(), "")

    def test_sha256_prefix_rejected(self):
        """``sha256:<hex>`` is a fast_gate-style prefix; the CLI takes bare hex."""
        with pytest.raises(click.BadParameter):
            _validate_content_hash_option(
                None, _FakeParam(), "sha256:" + ("a" * 64)
            )

    def test_whitespace_wrapped_rejected(self):
        """Leading/trailing whitespace MUST fail-loud per §5 — operator
        probably pasted with surrounding quotes or newlines; hint them."""
        with pytest.raises(click.BadParameter):
            _validate_content_hash_option(None, _FakeParam(), "  " + ("a" * 64))

    def test_error_message_cites_param_name(self):
        """BadParameter ctor receives param so Click renders the option name
        in the usage error. Locks the UX — operator sees 'Invalid value for
        --compatibility-fp: ...' not an opaque traceback."""
        param = _FakeParam()
        with pytest.raises(click.BadParameter) as exc_info:
            _validate_content_hash_option(None, param, "BAD")
        # param attribute reachable on the exception for Click renderer
        assert exc_info.value.param is param


# =============================================================================
# E2E: full Click machinery via CliRunner
# =============================================================================


class TestLedgerListCompatibilityFpE2E:
    """End-to-end: `hft-ops ledger list --compatibility-fp <val>` through
    the real Click dispatch. Locks the CLI-layer exit code + error
    rendering, which is what an operator actually sees."""

    def test_malformed_uppercase_exits_nonzero_with_usage_error(self):
        """Uppercase input → Click exits with code 2 (BadParameter convention)
        and stderr mentions the parameter name + required form."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["ledger", "list", "--compatibility-fp", "A" * 64],
            catch_exceptions=False,
        )
        assert result.exit_code != 0, "Malformed input must fail-loud, not silently filter"
        # Click renders BadParameter as "Error: Invalid value for '--compatibility-fp': ..."
        # Click 8.3+ merges stderr into result.output by default (`mix_stderr=True`);
        # older versions routed to result.stderr. Check both for cross-version
        # robustness.
        combined = result.output + (getattr(result, "stderr", "") or "")
        assert "--compatibility-fp" in combined
        assert "64 lowercase hex" in combined

    def test_malformed_truncated_exits_nonzero(self):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["ledger", "list", "--compatibility-fp", "abc123"],
            catch_exceptions=False,
        )
        assert result.exit_code != 0

    def test_option_absent_does_not_trigger_validator(self):
        """`hft-ops ledger list` (no --compatibility-fp) reaches command
        body without any validator rejection. The command may still fail
        downstream (no ledger dir) but NOT due to fingerprint validation."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["ledger", "list"],
            catch_exceptions=False,
        )
        # Regardless of exit code (which depends on ledger presence), the
        # failure surface MUST NOT mention --compatibility-fp as invalid.
        combined = result.output + (getattr(result, "stderr", "") or "")
        # If "Invalid value" appears, it must NOT be for --compatibility-fp
        # (it could legitimately appear for an unrelated missing arg).
        if "Invalid value" in combined:
            assert "--compatibility-fp" not in combined
