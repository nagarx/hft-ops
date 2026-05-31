"""M8 (2026-06-01): importing the ``hft_ops.feature_sets`` / ``hft_ops.ledger``
PACKAGES must NOT emit an hft-ops DeprecationWarning at import time.

Both packages' ``__init__.py`` historically re-exported THROUGH the local Phase-6
deprecation shims (``feature_sets/hashing.py``, ``ledger/experiment_record.py``),
whose module-level ``__getattr__`` fires a ``DeprecationWarning`` on first symbol
access. Merely importing the package therefore tripped the warning — and under a
strict ``-W error::DeprecationWarning`` CI/collection filter that becomes an
ImportError. The fix re-exports from the canonical ``hft_contracts`` homes in the
package ``__init__``, so the package import is warning-free while the shims keep
warning for any code that imports through them DIRECTLY.

Tested via a fresh subprocess interpreter: the shims latch one-warning-per-symbol-
per-process and ``sys.modules`` caches the already-imported package, so an in-process
re-import test would falsely pass. A fresh interpreter is exactly the CI-collection
scenario. The probe records all warnings and filters to hft-ops-originated
DeprecationWarnings, so unrelated third-party deprecations never flake the test.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# Record-all-warnings probe; exit 1 iff an hft-ops DeprecationWarning fired.
_IMPORT_PROBE = (
    "import warnings, sys\n"
    "with warnings.catch_warnings(record=True) as w:\n"
    "    warnings.simplefilter('always')\n"
    "    import {module}\n"
    "bad = [x for x in w if issubclass(x.category, DeprecationWarning) "
    "and 'hft_ops' in str(x.message)]\n"
    "sys.stderr.write('\\n'.join(str(x.message) for x in bad))\n"
    "sys.exit(1 if bad else 0)\n"
)


@pytest.mark.parametrize("module", ["hft_ops.feature_sets", "hft_ops.ledger"])
def test_package_import_emits_no_hft_ops_deprecationwarning(module: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE.format(module=module)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"importing {module} emitted an hft-ops DeprecationWarning at import time "
        f"(package __init__ routes through a deprecating shim):\n{proc.stderr}"
    )


def test_direct_shim_import_still_warns() -> None:
    """The ``__init__`` fix must NOT silence the shim for direct legacy importers —
    the migration signal must survive for code on the deprecated path."""
    probe = (
        "import warnings, sys\n"
        "with warnings.catch_warnings(record=True) as w:\n"
        "    warnings.simplefilter('always')\n"
        "    from hft_ops.feature_sets.hashing import compute_feature_set_hash\n"
        "bad = [x for x in w if issubclass(x.category, DeprecationWarning) "
        "and 'hft_ops' in str(x.message)]\n"
        "sys.exit(0 if bad else 1)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    assert proc.returncode == 0, (
        "direct import from the hft_ops.feature_sets.hashing shim must STILL emit "
        "its DeprecationWarning after the __init__ fix (migration signal preserved)"
    )
