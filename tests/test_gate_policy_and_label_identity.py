"""§13 gate policy must be declared, and every verdict must name its label.

Two 2026-08-15 operator rulings are locked here.

**R2 — no silent gate defaults.** Before this round, `on_fail`, `min_ic`,
`min_ic_count`, `min_return_std_bps`, `min_stability` and `primary_metric`
appeared in ZERO of the 58 manifests, so 189 ledger records ran the mandatory
gate on hard-coded fallbacks nobody chose. `on_fail` falling back to `"warn"`
is why four GATE:FAIL runs trained anyway.

**R3 — a verdict must name its dependent variable.** The gate scores the
on-disk label array; the trainer DERIVES labels at load time when
`labels.source` is `forward_prices`/`auto` and discards that array. Divergence
is RECORDED, never blocked — a deliberate label substitution is a legitimate
experiment (the NX1 arc did exactly that on purpose).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from hft_ops.config import OpsConfig
from hft_ops.manifest.loader import load_manifest
from hft_ops.manifest.schema import (
    GATE_POLICY_REQUIRED_KEYS,
    MissingGatePolicyError,
    require_gate_policy,
)
from hft_ops.paths import PipelinePaths

# Module-local fixture + manifest builder from the sibling gate-runner suite;
# importing them keeps ONE synthetic-export harness rather than forking a second.
from test_validation_stage import (  # noqa: F401 — `synthetic_ops_env` is a fixture
    _make_manifest,
    synthetic_ops_env,
)

from hft_ops.stages.validation import (
    _build_label_identity,
    _export_declares_forward_prices,
    _resolve_trainer_labels_config,
    _scored_label_identity,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"

_FULL_POLICY = {
    "enabled": True,
    "on_fail": "warn",
    "min_ic": 0.05,
    "min_ic_count": 2,
    "min_return_std_bps": 5.0,
    "min_stability": 2.0,
}


def _manifest(tmp_path: Path, validation: dict | None, name="m") -> Path:
    doc: dict = {
        "experiment": {"name": name},
        "pipeline_root": "..",
        "stages": {"extraction": {"enabled": False, "output_dir": "d"}},
    }
    if validation is not None:
        doc["stages"]["validation"] = validation
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p


# ---------------------------------------------------------------- R2: required


class TestGatePolicyRequired:
    def test_absent_validation_block_is_rejected(self, tmp_path):
        """An omitted block leaves `enabled` at True — the historical defect."""
        with pytest.raises(MissingGatePolicyError):
            load_manifest(_manifest(tmp_path, None))

    @pytest.mark.parametrize("dropped", sorted(GATE_POLICY_REQUIRED_KEYS["validation"]))
    def test_each_key_is_individually_required(self, tmp_path, dropped):
        policy = {k: v for k, v in _FULL_POLICY.items() if k != dropped}
        with pytest.raises(MissingGatePolicyError) as exc:
            load_manifest(_manifest(tmp_path, policy, name=f"drop_{dropped}"))
        assert dropped in str(exc.value)

    def test_message_is_actionable(self, tmp_path):
        """Names the key, the manifest, a sensible value, and the opt-out."""
        path = _manifest(tmp_path, {"enabled": True}, name="bare")
        with pytest.raises(MissingGatePolicyError) as exc:
            load_manifest(path)
        msg = str(exc.value)
        for key, (example, why) in GATE_POLICY_REQUIRED_KEYS["validation"].items():
            assert key in msg, f"missing key {key} not named"
            assert example in msg, f"no example value for {key}"
            assert why.split()[0] in msg
        assert "bare.yaml" in msg, "manifest not named"
        assert "enabled: false" in msg, "opt-out not offered"

    def test_all_missing_keys_reported_at_once(self, tmp_path):
        """One re-run per missing key would be hostile; report them together."""
        with pytest.raises(MissingGatePolicyError) as exc:
            load_manifest(_manifest(tmp_path, {"enabled": True}, name="all"))
        undeclared = str(exc.value).split("Undeclared key(s):")[1].split(".")[0]
        assert len(undeclared.split(",")) == len(
            GATE_POLICY_REQUIRED_KEYS["validation"]
        )

    def test_declared_policy_loads(self, tmp_path):
        stage = load_manifest(_manifest(tmp_path, dict(_FULL_POLICY))).stages.validation
        assert (stage.on_fail, stage.min_ic, stage.min_stability) == ("warn", 0.05, 2.0)

    def test_disabled_gate_needs_no_policy(self, tmp_path):
        """A gate that never runs applies no thresholds — demanding them is noise."""
        m = load_manifest(_manifest(tmp_path, {"enabled": False}, name="off"))
        assert m.stages.validation.enabled is False

    def test_post_training_gate_must_name_its_metric(self, tmp_path):
        doc = yaml.safe_load(_manifest(tmp_path, dict(_FULL_POLICY)).read_text())
        doc["stages"]["post_training_gate"] = {"enabled": True}
        p = tmp_path / "ptg.yaml"
        p.write_text(yaml.safe_dump(doc))
        with pytest.raises(MissingGatePolicyError) as exc:
            load_manifest(p)
        assert "primary_metric" in str(exc.value)

    def test_empty_stage_block_counts_as_undeclared(self):
        """`validation:` with no value parses to None, not to a policy."""
        with pytest.raises(MissingGatePolicyError):
            require_gate_policy("validation", None, enabled=True)

    def test_dataclass_roundtrip_is_exempt(self):
        """resolver.py feeds asdict(manifest): every key present, no false alarm."""
        require_gate_policy("validation", dict(_FULL_POLICY), enabled=True)

    def test_unknown_stage_name_is_a_noop(self):
        require_gate_policy("extraction", {}, enabled=True)


class TestShippedManifestsDeclareTheirPolicy:
    """Every manifest on disk must load — the round is not half-applied."""

    @pytest.mark.parametrize(
        "path",
        sorted(EXPERIMENTS.rglob("*.yaml")),
        ids=lambda p: p.name,
    )
    def test_manifest_loads(self, path):
        load_manifest(path)

    def test_no_enabled_gate_relies_on_a_default(self):
        required = GATE_POLICY_REQUIRED_KEYS["validation"]
        offenders = []
        for path in sorted(EXPERIMENTS.rglob("*.yaml")):
            doc = yaml.safe_load(path.read_text()) or {}
            raw = (doc.get("stages") or {}).get("validation") or {}
            if raw.get("enabled", True) and not all(k in raw for k in required):
                offenders.append(path.name)
        assert offenders == []


# ------------------------------------------------- R3: label identity recorded


@pytest.fixture()
def export(tmp_path):
    """Two days: one carrying BOTH label kinds, one carrying only regression."""
    train = tmp_path / "train"
    train.mkdir()
    for day, both in (("20250203", True), ("20250204", False)):
        np.save(train / f"{day}_sequences.npy", np.zeros((4, 2, 3), np.float32))
        np.save(train / f"{day}_regression_labels.npy", np.zeros((4, 3)))
        if both:
            np.save(train / f"{day}_labels.npy", np.zeros((4, 3), np.int8))
        (train / f"{day}_metadata.json").write_text(
            json.dumps({"forward_prices": {"exported": True}})
        )
    return tmp_path


class TestScoredLabelIdentity:
    def test_records_which_files_the_gate_read(self, export):
        got = _scored_label_identity(export, "train")
        assert got["resolved"] and got["n_files"] == 2
        # ExportLoader prefers labels.npy over regression_labels.npy
        # (hft_evaluator/data/loader.py:150-162) — record BOTH kinds so an
        # export carrying both cannot silently be scored on the wrong one.
        assert got["file_kinds"] == ["labels.npy", "regression_labels.npy"]

    def test_hash_is_mutation_sensitive(self, export):
        before = _scored_label_identity(export, "train")["content_hash"]
        np.save(export / "train" / "20250204_regression_labels.npy", np.ones((4, 3)))
        assert _scored_label_identity(export, "train")["content_hash"] != before

    def test_missing_export_degrades_and_never_raises(self, tmp_path):
        got = _scored_label_identity(tmp_path / "nope", "train")
        assert got["resolved"] is False and got["reason"]


class TestFittedLabelIdentity:
    def _cfg(self):
        return OpsConfig(paths=PipelinePaths(pipeline_root=REPO_ROOT))

    def test_inline_trainer_config_resolves_and_hashes(self):
        m = load_manifest(EXPERIMENTS / "sweeps/cycle6_r16a_point_vs_peak_H60.yaml")
        got = _resolve_trainer_labels_config(m, self._cfg())
        assert got["resolved"] and got["source"] == "forward_prices"
        assert len(got["label_strategy_hash"]) == 64

    def test_unresolvable_base_composition_says_so(self):
        """An honest 'unknown' beats a fabricated identity."""
        m = load_manifest(EXPERIMENTS / "nvda_tlob_h10_v1.yaml")
        got = _resolve_trainer_labels_config(m, self._cfg())
        assert got["resolved"] is False and "_base" in got["reason"]

    def test_overrides_win_over_the_base_block(self, tmp_path):
        m = load_manifest(EXPERIMENTS / "sweeps/cycle6_r16a_point_vs_peak_H60.yaml")
        base = _resolve_trainer_labels_config(m, self._cfg())
        m.stages.training.overrides["data.labels.return_type"] = "peak_return"
        after = _resolve_trainer_labels_config(m, self._cfg())
        assert after["return_type"] == "peak_return"
        assert after["label_strategy_hash"] != base["label_strategy_hash"]


class TestDivergenceVerdict:
    def _cfg(self):
        return OpsConfig(paths=PipelinePaths(pipeline_root=REPO_ROOT))

    def test_forward_prices_is_flagged_as_derived(self, export):
        m = load_manifest(EXPERIMENTS / "sweeps/cycle6_r16a_point_vs_peak_H60.yaml")
        got = _build_label_identity(m, self._cfg(), export, "train")
        assert got["divergence"] == "derived_at_load"
        assert got["scored"]["content_hash"]
        assert got["will_fit"]["label_strategy_hash"]

    def test_precomputed_source_is_not_flagged(self, export):
        m = load_manifest(EXPERIMENTS / "sweeps/cycle10_r19_multi_seed.yaml")
        got = _build_label_identity(m, self._cfg(), export, "train")
        assert got["divergence"] == "scores_fitted_array"

    def test_auto_depends_on_what_the_export_declares(self, export):
        """`auto` derives only when the export says forward_prices exist."""
        assert _export_declares_forward_prices(export, "train") is True
        for md in (export / "train").glob("*_metadata.json"):
            md.write_text(json.dumps({"forward_prices": {"exported": False}}))
        assert _export_declares_forward_prices(export, "train") is False

    def test_block_is_json_serializable(self, export):
        m = load_manifest(EXPERIMENTS / "sweeps/cycle6_r16a_point_vs_peak_H60.yaml")
        got = _build_label_identity(m, self._cfg(), export, "train")
        assert json.loads(json.dumps(got, default=str))["divergence"]

    def test_verdict_is_recorded_never_blocking(self, export):
        """R3: divergence must not be expressible as a failure."""
        m = load_manifest(EXPERIMENTS / "sweeps/cycle6_r16a_point_vs_peak_H60.yaml")
        got = _build_label_identity(m, self._cfg(), export, "train")
        assert "NEVER BLOCKING" in got["policy"]


class TestIdentityReachesTheArtifactAndTheLedger:
    """End-to-end: a real `ValidationRunner.run` must carry both identities.

    Reuses `test_validation_stage.py`'s synthetic-export harness — the gate
    genuinely runs (fast_gate over real arrays), so this proves the wiring,
    not just the helper.
    """

    def test_run_attaches_both_identities(self, synthetic_ops_env, tmp_path):
        from hft_ops.stages.base import StageStatus
        from hft_ops.stages.validation import ValidationRunner

        tmp_pipeline, rel = synthetic_ops_env
        manifest = load_manifest(_make_manifest(tmp_pipeline, rel, on_fail="warn"))
        ops = OpsConfig(
            paths=PipelinePaths(pipeline_root=tmp_pipeline),
            dry_run=False,
            verbose=False,
        )

        result = ValidationRunner().run(manifest, ops)
        assert result.status is StageStatus.COMPLETED

        identity = result.captured_metrics["gate_report"]["label_identity"]
        assert identity["scored"]["resolved"] is True, (
            "the gate must be able to say WHICH array it scored"
        )
        assert identity["divergence"] in {
            "derived_at_load",
            "scores_fitted_array",
            "unknown",
        }
        # Flat scalars so a ledger query can filter on divergence without
        # reaching into the nested report.
        assert "label_divergence" in result.captured_metrics
        assert "scored_label_content_hash" in result.captured_metrics
        assert "fitted_label_strategy_hash" in result.captured_metrics

        # The on-disk artifact and the in-memory record must agree — a
        # gate_report.json without the identity would be the same
        # derive-then-discard defect this whole block exists to expose.
        on_disk = json.loads(
            Path(result.captured_metrics["gate_report_path"]).read_text()
        )
        assert on_disk["label_identity"]["divergence"] == identity["divergence"]
        assert on_disk["verdict"] == result.captured_metrics["validation_verdict"]

    def test_identity_failure_cannot_fail_the_stage(
        self, synthetic_ops_env, monkeypatch
    ):
        """R3 is unconditional: if identity recording breaks, the gate stands."""
        from hft_ops.stages import validation as vmod
        from hft_ops.stages.base import StageStatus

        tmp_pipeline, rel = synthetic_ops_env
        manifest = load_manifest(_make_manifest(tmp_pipeline, rel, on_fail="warn"))
        ops = OpsConfig(
            paths=PipelinePaths(pipeline_root=tmp_pipeline),
            dry_run=False,
            verbose=False,
        )

        def _boom(*a, **k):
            raise RuntimeError("identity subsystem exploded")

        monkeypatch.setattr(vmod, "_build_label_identity", _boom)
        result = vmod.ValidationRunner().run(manifest, ops)
        assert result.status is StageStatus.COMPLETED
        assert result.captured_metrics["validation_verdict"] == "PASS"
        assert "label_identity" not in result.captured_metrics["gate_report"]


class TestReturnTypeDiscrimination:
    """FINDING-164: `divergence` alone cannot tell the 64 from the 24.

    `divergence` is keyed on `labels.source` ALONE, so it reads
    `derived_at_load` for ALL 88 passing ledger records -- including the 24
    whose fitted return type is the one the export already holds. A warning
    wrong on 27% of its own subject is how a gate earns being routed around.

    These lock the discriminating comparison added 2026-08-17. It is ADDITIVE:
    `divergence` keeps its old values so no existing consumer changes
    behaviour.
    """

    def test_conventions_are_folded_across_the_language_boundary(self):
        """The exporter writes PascalCase, the trainer config snake_case.

        Comparing them raw reports EVERY record as divergent -- the same class
        of error as comparing a name where an identity was meant.
        """
        from hft_ops.stages.validation import _normalize_return_type

        assert _normalize_return_type("SmoothedReturn") == _normalize_return_type(
            "smoothed_return"
        )
        assert _normalize_return_type("PointReturn") == _normalize_return_type(
            "point_return"
        )
        assert _normalize_return_type("PointReturn") != _normalize_return_type(
            "smoothed_return"
        )

    @pytest.mark.parametrize("bad", [None, "", "   ", 3, [], {}])
    def test_a_missing_return_type_is_unknown_not_agreement(self, bad):
        """An absent declaration must never read as a match."""
        from hft_ops.stages.validation import _normalize_return_type

        assert _normalize_return_type(bad) is None

    def test_export_declaration_is_read_from_the_explicit_field(self, tmp_path):
        """Read `labeling.return_type`, NOT the prose in label_encoding.

        The field is present on all 234 regression-export days surveyed
        2026-08-17. Parsing the description string instead would be a
        string-shaped guess at an identity.
        """
        from hft_ops.stages.validation import _export_declared_return_type

        split = tmp_path / "train"
        split.mkdir()
        (split / "2025-02-03_metadata.json").write_text(
            json.dumps(
                {
                    "labeling": {
                        "return_type": "PointReturn",
                        "label_encoding": {
                            "description": "SmoothedReturn forward return in bps"
                        },
                    }
                }
            )
        )
        # The explicit field wins over the contradictory prose.
        assert _export_declared_return_type(tmp_path, "train") == "pointreturn"

    def test_unreadable_export_is_unknown_not_agreement(self, tmp_path):
        from hft_ops.stages.validation import _export_declared_return_type

        assert _export_declared_return_type(tmp_path, "train") is None
        split = tmp_path / "train"
        split.mkdir()
        (split / "2025-02-03_metadata.json").write_text("{not json")
        assert _export_declared_return_type(tmp_path, "train") is None

    def test_differ_is_conclusive_and_match_is_only_necessary(self):
        """Lock the RUNTIME semantics string, not the source text.

        The first version of this test asserted the words appeared in the
        function's SOURCE — and passed even after they were deleted from the
        emitted string, because the same words sat in a nearby comment. Caught
        by mutation: 3 of 4 mutants went red, this one stayed green.

        Acting on 'match' as if it were proof is the next version of the defect
        FINDING-164 records, so the asymmetry must reach the consumer.
        """
        from hft_ops.stages.validation import RETURN_TYPE_MATCH_SEMANTICS as S

        assert "CONCLUSIVE" in S, "'differ' must be stated as conclusive"
        assert "NOT SUFFICIENT" in S, "'match' must be stated as insufficient"
        assert "unknown" in S, "the third state must be documented too"
