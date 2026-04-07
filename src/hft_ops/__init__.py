"""
hft-ops: Central experiment orchestrator for the HFT pipeline.

Defines, validates, runs, tracks, and compares experiments across all
pipeline modules from a single YAML manifest.

Usage:
    hft-ops run experiments/my_experiment.yaml
    hft-ops validate experiments/my_experiment.yaml
    hft-ops compare --metric macro_f1 --sort desc
    hft-ops ledger list
"""

__version__ = "0.2.0"
