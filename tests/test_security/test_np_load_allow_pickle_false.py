"""#PY-291 lock test — every ``np.load()`` must pass ``allow_pickle=False``.

Sister of FIND-110 lock-test at ``lob-backtester/tests/test_security/`` +
trainer lock-test at ``lob-model-trainer/tests/test_security/``. Closes the
RCE-via-malicious-NPY class for the hft-ops orchestrator surface (3 src
callsites at ``ledger/r16c_analysis.py`` + ``ledger/statistical_compare.py``).

See ``CROSS_PIPELINE_VALIDATION_FINDINGS_2026_05_16.md`` §3 #PY-291.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NP_LOAD_RE = re.compile(r"np\.load\s*\(")
ALLOW_PICKLE_FALSE_RE = re.compile(r"allow_pickle\s*=\s*False")


def _extract_call_span(text: str, open_paren_idx: int) -> str:
    """Return the call's argument span (handles nested parens)."""
    depth = 0
    end = open_paren_idx
    for i in range(open_paren_idx, len(text)):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    return text[open_paren_idx : end + 1]


class TestPy291AllowPickleFalseLock:
    """#PY-291 lock: every ``np.load()`` must pass ``allow_pickle=False``."""

    def test_every_np_load_passes_allow_pickle_false(self):
        """No ``np.load()`` callsite in src/, tests/, scripts/ may omit
        ``allow_pickle=False``.

        Closes #PY-291 RCE-via-malicious-NPY hazard surfaced by Wave 2-C
        hidden findings hunt 2026-05-16. If this test fails, the listed
        offenders MUST be hardened before merge.
        """
        offenders = []
        for sub in ("src", "tests", "scripts"):
            base = REPO_ROOT / sub
            if not base.exists():
                continue
            for py in base.rglob("*.py"):
                if py.name == "test_np_load_allow_pickle_false.py":
                    continue
                text = py.read_text()
                for m in NP_LOAD_RE.finditer(text):
                    span = _extract_call_span(text, m.end() - 1)
                    if not ALLOW_PICKLE_FALSE_RE.search(span):
                        line = text[: m.start()].count("\n") + 1
                        offenders.append(f"{py.relative_to(REPO_ROOT)}:{line}")
        assert not offenders, (
            "#PY-291 lock: every np.load() callsite must pass "
            "allow_pickle=False (prevents pickle-RCE on malicious .npy "
            "files; hft-rules §8). Offenders:\n  " + "\n  ".join(offenders)
        )

    def test_no_aliased_numpy_load_imports(self):
        """Defensive lock against aliased imports that bypass np.load scan."""
        forbidden_re = re.compile(
            r"from\s+numpy\s+import\s+.*\bload\b|import\s+numpy\.load\b"
        )
        offenders = []
        for sub in ("src", "tests", "scripts"):
            base = REPO_ROOT / sub
            if not base.exists():
                continue
            for py in base.rglob("*.py"):
                if py.name == "test_np_load_allow_pickle_false.py":
                    continue
                text = py.read_text()
                for m in forbidden_re.finditer(text):
                    line = text[: m.start()].count("\n") + 1
                    offenders.append(
                        f"{py.relative_to(REPO_ROOT)}:{line} ({m.group(0)})"
                    )
        assert not offenders, (
            "Aliased numpy.load imports defeat the np.load() regression-lock "
            "test. Use 'import numpy as np; np.load(...)' instead. Offenders:\n  "
            + "\n  ".join(offenders)
        )
