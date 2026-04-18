# DATA PREP UTILITY -- not an experiment
"""Phase 7 Stage 7.1 (2026-04-18) — migrate `feature_preset:` usage to the
Phase 4 FeatureSet registry before the 2026-08-15 ImportError deadline.

Creates 3 production FeatureSet JSONs at ``contracts/feature_sets/`` that mirror
the existing trainer preset definitions (``lobtrainer.constants.feature_presets``):

    nvda_short_term_40_src128_v1.json    (PRESET_SHORT_TERM_40 on 128-feat exports)
    nvda_short_term_40_src116_v1.json    (PRESET_SHORT_TERM_40 on 116-feat exports)
    nvda_analysis_ready_119_src128_v1.json (PRESET_ANALYSIS_READY_128 on 128-feat)

After this script runs, the 8 trainer + hft-ops manifests using ``feature_preset:``
can be migrated via config edits to reference the new FeatureSet names.

Design choices
--------------
* **Hand-curated, NOT evaluator-derived** — indices are literal mirrors of
  ``PRESET_SHORT_TERM_40`` + ``PRESET_ANALYSIS_READY_128`` as they exist today.
  Statistical rigor (IC gate, dCor filter, etc.) is left to FUTURE FeatureSets
  produced via ``hft-ops evaluate --save-feature-set``. Phase 7.1 is a
  back-compat MIGRATION, not a feature-selection re-derivation.
* **Two SFC-variant FeatureSets for SHORT_TERM_40** — one trainer config
  (``nvda_short_term_hmhp_v1.yaml``) declares ``feature_count: 116`` while
  others use 128. Resolver enforces strict equality on ``source_feature_count``,
  so each SFC needs its own FeatureSet artifact (same indices, different
  ``content_hash`` due to SFC being in the hash input).
* **Idempotent** — ``write_feature_set`` has refuse-overwrite + identity-
  match-no-op semantics. Re-running this script is safe: if the JSON already
  exists with the correct content_hash, it is a no-op; if the hash changed,
  it raises ``FeatureSetExists`` (intentional: content_hash drift between
  run-A and run-B indicates a preset definition change and demands a
  version-bumped new FeatureSet).
* **Names embed SFC** — ``_src128_v1`` / ``_src116_v1`` suffix makes the
  SFC explicit in every consumer manifest. Future FeatureSets can follow
  the same convention.

Uses ``hft_ops.feature_sets.writer`` for atomic-write guarantees; only
hft-contracts + hft-ops dependencies (no trainer import). Preset indices
are hardcoded here (verified against trainer source by companion test at
``lob-model-trainer/tests/test_feature_preset_migration.py``).

Usage
-----

    cd hft-ops
    .venv/bin/python scripts/migrate_feature_presets_to_registry.py \\
        --registry-dir ../contracts/feature_sets

By default ``--registry-dir`` resolves to ``<monorepo_root>/contracts/feature_sets``
(walks up from script location).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

from hft_contracts import FeatureIndex, ExperimentalFeatureIndex, SCHEMA_VERSION
from hft_contracts.feature_sets import (
    FeatureSet,
    FeatureSetAppliesTo,
    FeatureSetProducedBy,
)
from hft_ops.feature_sets.writer import write_feature_set, FeatureSetExists


# -----------------------------------------------------------------------------
# Preset index lists — hand-curated to match trainer source of truth.
# VERIFIED against `lob-model-trainer/src/lobtrainer/constants/feature_presets.py`
# via `lob-model-trainer/tests/test_feature_preset_migration.py::test_indices_match_trainer_presets`.
# Do NOT edit these without re-running that parity test.
# -----------------------------------------------------------------------------

# Mirror of PRESET_SHORT_TERM_40 (40 indices, L1-L2 book + derived + flow +
# signals + safety gates + institutional patience; optimized for H10/H20).
PRESET_SHORT_TERM_40_INDICES: Tuple[int, ...] = (
    # L1-L2 Order Book (8)
    0, 1, 10, 11, 20, 21, 30, 31,
    # Derived (7): mid, spread, spread_bps, bid_vol, ask_vol, vol_imbalance, weighted_mid
    40, 41, 42, 43, 44, 45, 46,
    # Order flow dynamics (9): add_{bid,ask}, cancel_{bid,ask}, trade_{bid,ask}, net_{order,cancel,trade}
    48, 49, 50, 51, 52, 53, 54, 55, 56,
    # Flow indicators (3): aggressive_order_ratio, order_flow_volatility, flow_regime
    57, 58, 59,
    # Trading signals (8): true_ofi, depth_norm_ofi, executed_pressure, signed_mp_delta_bps,
    #   trade_asymmetry, cancel_asymmetry, fragility_score, depth_asymmetry
    84, 85, 86, 87, 88, 89, 90, 91,
    # Safety gates (3): book_valid, time_regime, mbo_ready
    92, 93, 94,
    # Institutional patience (2): fill_patience_{bid,ask}
    104, 105,
)
assert len(PRESET_SHORT_TERM_40_INDICES) == 40
assert len(set(PRESET_SHORT_TERM_40_INDICES)) == 40  # no duplicates
assert min(PRESET_SHORT_TERM_40_INDICES) == 0
assert max(PRESET_SHORT_TERM_40_INDICES) == 105

# Mirror of PRESET_ANALYSIS_READY_128 — all 128 features minus 9 dead/constant.
_DEAD_FEATURES = frozenset({68, 69, 76, 77, 92, 94, 96, 97, 102})
PRESET_ANALYSIS_READY_119_INDICES: Tuple[int, ...] = tuple(
    i for i in range(128) if i not in _DEAD_FEATURES
)
assert len(PRESET_ANALYSIS_READY_119_INDICES) == 119


def name_for_index(idx: int) -> str:
    """Look up the feature name from the hft_contracts enums.

    Returns the enum-member name in lowercase. Falls back to ``feature_{idx}``
    if the index is not in either enum (should not occur for valid presets).
    """
    for member in FeatureIndex:
        if member.value == idx:
            return member.name.lower()
    for member in ExperimentalFeatureIndex:
        if member.value == idx:
            return member.name.lower()
    return f"feature_{idx}"


def _locate_monorepo_root(script_path: Path) -> Path:
    """Walk up from this script to the HFT-pipeline-v2 monorepo root.

    The root is identified by the presence of ``contracts/pipeline_contract.toml``.
    """
    for parent in script_path.resolve().parents:
        if (parent / "contracts" / "pipeline_contract.toml").exists():
            return parent
    raise RuntimeError(
        f"Could not locate HFT-pipeline-v2 root from {script_path}; "
        f"expected `contracts/pipeline_contract.toml` in some parent directory."
    )


def build_feature_set(
    *,
    name: str,
    indices: Tuple[int, ...],
    source_feature_count: int,
    applies_to_horizons: Tuple[int, ...],
    description: str,
    mirrors_preset: str,
) -> FeatureSet:
    """Construct a hand-curated FeatureSet with explicit provenance.

    Uses ``FeatureSet.build`` which auto-computes the content_hash over the
    PRODUCT fields (indices, SFC, contract_version).
    """
    names = tuple(name_for_index(i) for i in indices)
    return FeatureSet.build(
        name=name,
        feature_indices=indices,
        feature_names=names,
        source_feature_count=source_feature_count,
        contract_version=SCHEMA_VERSION,
        applies_to=FeatureSetAppliesTo(
            assets=("NVDA",),
            horizons=applies_to_horizons,
        ),
        produced_by=FeatureSetProducedBy(
            tool="hft-ops/scripts/migrate_feature_presets_to_registry.py",
            tool_version="phase7.1-stage1",
            config_path="lob-model-trainer/src/lobtrainer/constants/feature_presets.py",
            # Hand-curated migration: no config file drove this selection.
            # Traceability is via the mirrors_preset + the companion
            # parity test that locks indices to the trainer source.
            config_hash="",
            source_profile_hash="",
            data_export="",
            data_dir_hash="",
        ),
        criteria={
            "source": "hand-curated-from-feature_preset",
            "mirrors": f"lobtrainer.constants.feature_presets.{mirrors_preset}",
            "migration_context": (
                "Phase 7 Stage 7.1 (2026-04-18) — mirror of deprecated "
                "`feature_preset` definition. DeprecationWarning emitted "
                "since 2026-04-15; ImportError scheduled for 2026-08-15. "
                "See PHASE7_ROADMAP.md §7.1 for migration rationale."
            ),
        },
        criteria_schema_version="1.0",
        description=description,
        notes=(
            f"source_feature_count={source_feature_count} variant of "
            f"{mirrors_preset}. Same indices as the SFC-sibling "
            f"FeatureSet(s); content_hash differs because SFC is part of "
            f"the PRODUCT-hash input (indices, SFC, contract_version)."
        ),
        created_at=datetime.now(timezone.utc).isoformat(),
        created_by="phase7.1-migration",
    )


def migrate(registry_dir: Path, *, force: bool = False) -> int:
    """Write 3 FeatureSet JSONs. Returns the number successfully written.

    Idempotent: files with matching content_hash are no-ops; files with
    mismatched hash raise ``FeatureSetExists`` unless ``force=True``.
    """
    registry_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    # FeatureSet 1: SHORT_TERM_40 on 128-feature exports
    fs = build_feature_set(
        name="nvda_short_term_40_src128_v1",
        indices=PRESET_SHORT_TERM_40_INDICES,
        source_feature_count=128,
        applies_to_horizons=(10, 20),
        description=(
            "NVDA short-term 40-feature subset (L1-L2 book + derived + "
            "order flow + flow indicators + primary signals + safety gates "
            "+ institutional patience) for H10/H20 prediction on 128-feature "
            "MBO exports. Phase 7 Stage 7.1 migration from "
            "`feature_preset: short_term_40` (deprecated 2026-04-15, "
            "ImportError 2026-08-15)."
        ),
        mirrors_preset="PRESET_SHORT_TERM_40",
    )
    print(f"Building {fs.name}: {len(fs.feature_indices)} indices, "
          f"sfc={fs.source_feature_count}, hash={fs.content_hash[:12]}...")
    try:
        write_feature_set(
            registry_dir / f"{fs.name}.json",
            fs,
            force=force,
        )
        written += 1
        print(f"  WROTE {fs.name}.json")
    except FeatureSetExists as e:
        if "identical" in str(e).lower() or "idempotent" in str(e).lower():
            print(f"  SKIPPED {fs.name}.json (identical content already on disk)")
        else:
            raise

    # FeatureSet 2: SHORT_TERM_40 on 116-feature exports (legacy variant)
    fs = build_feature_set(
        name="nvda_short_term_40_src116_v1",
        indices=PRESET_SHORT_TERM_40_INDICES,
        source_feature_count=116,
        applies_to_horizons=(10, 20),
        description=(
            "NVDA short-term 40-feature subset on 116-feature MBO exports "
            "(legacy export shape used by pre-Phase-6 HMHP configs). Same "
            "indices as `nvda_short_term_40_src128_v1` but different "
            "content_hash due to source_feature_count=116. Phase 7 Stage "
            "7.1 migration target for `nvda_short_term_hmhp_v1.yaml`."
        ),
        mirrors_preset="PRESET_SHORT_TERM_40",
    )
    print(f"Building {fs.name}: {len(fs.feature_indices)} indices, "
          f"sfc={fs.source_feature_count}, hash={fs.content_hash[:12]}...")
    try:
        write_feature_set(
            registry_dir / f"{fs.name}.json",
            fs,
            force=force,
        )
        written += 1
        print(f"  WROTE {fs.name}.json")
    except FeatureSetExists as e:
        if "identical" in str(e).lower() or "idempotent" in str(e).lower():
            print(f"  SKIPPED {fs.name}.json (identical content already on disk)")
        else:
            raise

    # FeatureSet 3: ANALYSIS_READY_128 (all 128 minus 9 dead = 119 features)
    fs = build_feature_set(
        name="nvda_analysis_ready_119_src128_v1",
        indices=PRESET_ANALYSIS_READY_119_INDICES,
        source_feature_count=128,
        applies_to_horizons=(10, 20, 50, 60, 100, 200, 300),
        description=(
            "NVDA general-purpose 119-feature set (all 128 stable + "
            "experimental minus 9 dead/constant features: {68, 69, 76, "
            "77, 92, 94, 96, 97, 102}). Used as the primary training "
            "preset for 128-feature exports. Phase 7 Stage 7.1 migration "
            "from `feature_preset: analysis_ready_128` (deprecated "
            "2026-04-15, ImportError 2026-08-15)."
        ),
        mirrors_preset="PRESET_ANALYSIS_READY_128",
    )
    print(f"Building {fs.name}: {len(fs.feature_indices)} indices, "
          f"sfc={fs.source_feature_count}, hash={fs.content_hash[:12]}...")
    try:
        write_feature_set(
            registry_dir / f"{fs.name}.json",
            fs,
            force=force,
        )
        written += 1
        print(f"  WROTE {fs.name}.json")
    except FeatureSetExists as e:
        if "identical" in str(e).lower() or "idempotent" in str(e).lower():
            print(f"  SKIPPED {fs.name}.json (identical content already on disk)")
        else:
            raise

    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=None,
        help="Directory to write FeatureSet JSONs. Default: "
        "<monorepo_root>/contracts/feature_sets/",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing FeatureSets whose content_hash differs "
        "(dangerous: normally you want to bump the version name instead).",
    )
    args = parser.parse_args()

    script_path = Path(__file__)
    if args.registry_dir is None:
        monorepo_root = _locate_monorepo_root(script_path)
        args.registry_dir = monorepo_root / "contracts" / "feature_sets"

    print(f"Migration script: {script_path.name}")
    print(f"Registry dir:     {args.registry_dir}")
    print(f"Force:            {args.force}")
    print()

    written = migrate(args.registry_dir, force=args.force)
    print()
    print(f"Phase 7 Stage 7.1 migration complete: {written}/3 FeatureSet(s) "
          f"written (or skipped as idempotent no-ops).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
