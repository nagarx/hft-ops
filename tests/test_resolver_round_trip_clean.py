"""R3 lock: the resolver's 2nd ``Stages`` build site is typo-proof by construction.

``resolve_variables_in_manifest`` rebuilds ``Stages`` via the same ``_build_*``
helpers as ``load_manifest``, but it is fed ``asdict(manifest)`` whose stage
keys are already the 8 clean field names. So the H1 unknown-stage RAISE and the
H2 per-stage / top-level unknown-key WARNs (which live in ``load_manifest``)
cannot fire here. This locks that invariant: a future refactor that fed a
raw-YAML path into the resolver — reopening the silent typo gap — breaks these
tests.

Provenance: VALIDATION_AND_DESIGN_2026_05_30.md §12 Step 8 (R3).
"""

import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from hft_ops.manifest._field_introspection import stage_names
from hft_ops.manifest.loader import load_manifest
from hft_ops.manifest.resolver import (
    VarResolutionContext,
    resolve_variables_in_manifest,
)


def _make_manifest(tmp_path: Path):
    p = tmp_path / "m.yaml"
    p.write_text(
        """
experiment:
  name: r3_round_trip
  contract_version: "2.2"
pipeline_root: "."
stages:
  extraction:
    enabled: true
    output_dir: data/exports/e5_60s
  training:
    enabled: true
    trainer_config:
      model:
        model_type: tlob
  backtesting:
    enabled: true
    data_dir: "${stages.extraction.output_dir}"
"""
    )
    return load_manifest(p)


class TestResolverRoundTripClean:
    def test_resolver_emits_no_unknown_key_warnings(self, tmp_path: Path):
        m = _make_manifest(tmp_path)  # load-time warnings (none) fire OUTSIDE the block
        ctx = VarResolutionContext(now=datetime(2026, 4, 17, tzinfo=timezone.utc))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resolve_variables_in_manifest(m, ctx)
        unknown_key_warns = [
            w
            for w in caught
            if w.category is RuntimeWarning
            and (
                "silently dropping unknown" in str(w.message)
                or "unknown top-level keys" in str(w.message)
            )
        ]
        assert unknown_key_warns == [], (
            "resolver round-trip emitted unknown-key warnings (asdict should "
            "only produce clean field keys): "
            f"{[str(w.message) for w in unknown_key_warns]}"
        )

    def test_resolver_drops_no_stage_fields(self, tmp_path: Path):
        m = _make_manifest(tmp_path)
        ctx = VarResolutionContext(now=datetime(2026, 4, 17, tzinfo=timezone.utc))
        rebuilt = resolve_variables_in_manifest(m, ctx)
        orig = asdict(m.stages)
        new = asdict(rebuilt.stages)
        assert set(orig) == set(new) == set(stage_names()), (
            "stage set changed across resolver round-trip"
        )
        for sname in orig:
            assert set(orig[sname]) == set(new[sname]), (
                f"stage {sname!r} field set changed across resolver round-trip: "
                f"orig={sorted(orig[sname])} new={sorted(new[sname])}"
            )

    def test_resolver_still_resolves_cross_stage_ref(self, tmp_path: Path):
        # Sanity: the resolver's actual job (${...} re-resolution) still works.
        m = _make_manifest(tmp_path)
        ctx = VarResolutionContext(now=datetime(2026, 4, 17, tzinfo=timezone.utc))
        rebuilt = resolve_variables_in_manifest(m, ctx)
        assert rebuilt.stages.backtesting.data_dir == "data/exports/e5_60s"
