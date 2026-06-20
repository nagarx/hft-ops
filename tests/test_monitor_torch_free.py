"""The torch-free invariant for hft_ops.monitor (F5).

Mirrors test_contract_preflight.py's two-way guard: a static AST scan of EVERY
monitor module (auto-extends as modules are added) + a runtime sys.modules
sentinel. hft-ops is torch-free by design; the monitor reads ledger + discovery
JSONs and must never import torch / lobmodels / lobtrainer at module scope.
"""

import ast
import subprocess
import sys
from pathlib import Path

import hft_ops.monitor as monitor_pkg

_FORBIDDEN = ("torch", "lobmodels", "lob_models", "lobtrainer")


def _monitor_module_files():
    pkg_dir = Path(monitor_pkg.__file__).parent
    return sorted(pkg_dir.glob("*.py"))


def test_monitor_modules_are_torch_free_ast():
    """Static AST: no module-scope import of a forbidden package in any monitor file."""
    offenders = {}
    for src_path in _monitor_module_files():
        tree = ast.parse(src_path.read_text())
        imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        bad = [i for i in imports if i and any(i.startswith(p) for p in _FORBIDDEN)]
        if bad:
            offenders[src_path.name] = bad
    assert not offenders, (
        "hft_ops.monitor must be torch-free at module scope (root CLAUDE.md "
        f"§Module Technical Map). Offending imports: {offenders}. Gate any "
        "lazy import inside a function body or behind `if TYPE_CHECKING:`."
    )


def test_monitor_runtime_torch_free():
    """Runtime sentinel — defense-in-depth for the AST scan (catches dynamic imports)."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import hft_ops.monitor; "
            "bad=[m for m in sys.modules if m.startswith(('torch','lobmodels','lobtrainer'))]; "
            "assert not bad, f'torch-free violation: {bad}'; print('TORCH_FREE_OK')",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Runtime torch-free sentinel FAILED.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "TORCH_FREE_OK" in result.stdout
