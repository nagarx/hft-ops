"""Phase 4 Batch 4c.3 Enhancement B: LRU cache for FeatureSet resolution.

Locks:
1. 150× calls with the SAME (registry, name) hit the cache and avoid
   repeated disk I/O — `FeatureSetRegistry.get()` is called ONCE.
2. `cache_clear()` actually invalidates the cache (test-isolation primitive).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hft_ops.ledger.dedup import _cached_resolve_feature_set_indices
from hft_ops.feature_sets.schema import FeatureSet, FeatureSetAppliesTo, FeatureSetProducedBy
from hft_ops.feature_sets.writer import write_feature_set


@pytest.fixture
def registry(tmp_path: Path) -> Path:
    r = tmp_path / "feature_sets"
    r.mkdir()
    fs = FeatureSet.build(
        name="cached_v1", feature_indices=[5, 12, 84],
        feature_names=["f_5", "f_12", "f_84"],
        source_feature_count=98, contract_version="2.2",
        applies_to=FeatureSetAppliesTo(assets=("NVDA",), horizons=(10,)),
        produced_by=FeatureSetProducedBy(
            tool="t", tool_version="0", config_path="x",
            config_hash="a" * 64, source_profile_hash="b" * 64,
            data_export="d", data_dir_hash="c" * 64,
        ),
        criteria={}, criteria_schema_version="1.0",
    )
    write_feature_set(r / "cached_v1.json", fs)
    return r


class TestLRUCacheHits:
    def test_repeated_calls_use_cache(self, registry, monkeypatch):
        """Call _cached_resolve_feature_set_indices 150× with same args;
        `FeatureSetRegistry.get()` must be invoked ONCE (then cache hits)."""
        _cached_resolve_feature_set_indices.cache_clear()

        call_count = {"n": 0}
        import hft_ops.feature_sets.registry as registry_mod
        real_get = registry_mod.FeatureSetRegistry.get

        def counting_get(self, name, *, verify=True):
            call_count["n"] += 1
            return real_get(self, name, verify=verify)

        monkeypatch.setattr(registry_mod.FeatureSetRegistry, "get", counting_get)

        for _ in range(150):
            _ = _cached_resolve_feature_set_indices(str(registry), "cached_v1")

        # Expect exactly 1 call (first hit miss, rest cache)
        assert call_count["n"] == 1, (
            f"LRU cache miss: FeatureSetRegistry.get() called "
            f"{call_count['n']} times for identical args (expected 1). "
            f"150-point sweeps would pay 150× disk I/O."
        )

    def test_cache_clear_invalidates(self, registry):
        """cache_clear() is a public test-isolation primitive."""
        _cached_resolve_feature_set_indices.cache_clear()
        info0 = _cached_resolve_feature_set_indices.cache_info()
        assert info0.currsize == 0

        _ = _cached_resolve_feature_set_indices(str(registry), "cached_v1")
        info1 = _cached_resolve_feature_set_indices.cache_info()
        assert info1.currsize == 1

        _cached_resolve_feature_set_indices.cache_clear()
        info2 = _cached_resolve_feature_set_indices.cache_info()
        assert info2.currsize == 0, "cache_clear() must empty the LRU"
