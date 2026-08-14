# NX1 clean-label re-derivation — evidence snapshot, 2026-08-14

**Why this exists.** On 2026-08-14 a preservation sweep measured these six files as
`nlink=1` with **zero git objects in any of the 30 object stores on this machine**:

```
hft-ops/ledger/records/NX1_TLOB_128feat_TWAP_H10_v3p0_*.json      9,069 B   stores=NONE
hft-ops/ledger/records/NX1_HMHP_R_128feat_TWAP_H10_v3p0_*.json   11,192 B   stores=NONE
hft-ops/ledger/records/NX1_E5_TLOB_98feat_TWAP_H10_v3p0_*.json    7,399 B   stores=NONE
lob-model-trainer/outputs/experiments/nx1_*/test_metrics.json    3 files    stores=NONE
```

Of **189** records in `ledger/records/`, only these three were uncovered — the other 186 are
content-identical to the tracked `ledger/archive/2026-07-07-snapshot/`. These three postdate
that snapshot and were never captured.

**What they are.** The measured basis of **EXPERIMENT-069/070/071**, the clean-label
re-derivation that removed the evidential basis from three Phase-1 architecture claims. The
records carry `test_zero_skill_r2`, `test_excess_r2` and the training metrics; the
`test_metrics.json` files are the trainer-side originals.

⚠️ **A RULE IN FORCE POINTED AT THEM BY PATH.** `hft-rules §13` cites
`hft-ops/ledger/records/NX1_*.json::training_metrics.test_zero_skill_r2` verbatim as the
authority for the in-sample-floor correction. A governing rule was resting on files that one
`rm -rf` would have removed from existence.

**Snapshot, not a move.** The originals stay at their live paths (archive, never delete).
Verified byte-identical by sha256, 6 of 6, at copy time.

⚠️ **Read as evidence, not as conclusions.** The adjudicated result lives in the wiki register
(EXPERIMENT-069/070/071) and in root `CLAUDE.md`. Note in particular that `excess_r2` is
recorded there as **NOT an identified estimand** — these records carry the raw quantity, and
the register carries the caveats that must travel with it.
