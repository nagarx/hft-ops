"""Phase 8A.0+ — hft-ops scheduler package.

Homes content-addressed extraction cache (8A.0) and future parallel-
execution primitives (8A.1 — ProcessPoolExecutor + GPUSemaphore +
signal handling). Deliberately separate from `hft_ops.stages.*` because
scheduling is a CROSS-stage concern (cache reuse spans grid points;
parallelism dispatches stages).
"""
