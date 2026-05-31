"""R4 / Step 7: ``--stages`` CLI fail-loud validation (symmetric to H1).

A typo'd ``--stages`` value (e.g. ``--stages trainning``) must raise
``click.BadParameter`` instead of silently matching no stage and skipping it —
the same stage-name drift footgun H1 closes on the manifest surface, here on
the CLI surface.

Provenance: VALIDATION_AND_DESIGN_2026_05_30.md §12 Step 7 (R4 / Component 7).
"""

import click
import pytest

from hft_ops.cli import _parse_and_validate_stages
from hft_ops.manifest._field_introspection import stage_names


class TestParseAndValidateStages:
    def test_none_and_empty_return_none(self):
        assert _parse_and_validate_stages(None) is None
        assert _parse_and_validate_stages("") is None

    def test_valid_single_stage(self):
        assert _parse_and_validate_stages("training") == {"training"}

    def test_valid_multiple_stages(self):
        assert _parse_and_validate_stages("extraction,training") == {
            "extraction",
            "training",
        }

    def test_whitespace_is_stripped(self):
        assert _parse_and_validate_stages("extraction, training") == {
            "extraction",
            "training",
        }

    def test_unknown_stage_raises_badparameter(self):
        with pytest.raises(click.BadParameter) as exc:
            _parse_and_validate_stages("trainning")  # the classic typo
        assert "trainning" in str(exc.value)

    def test_error_lists_valid_stages(self):
        with pytest.raises(click.BadParameter) as exc:
            _parse_and_validate_stages("bogus")
        msg = str(exc.value)
        assert "training" in msg and "backtesting" in msg

    def test_mixed_valid_and_invalid_raises_citing_invalid(self):
        with pytest.raises(click.BadParameter) as exc:
            _parse_and_validate_stages("training,bogus")
        assert "bogus" in str(exc.value)

    def test_all_canonical_stages_valid(self):
        csv = ",".join(sorted(stage_names()))
        assert _parse_and_validate_stages(csv) == set(stage_names())
