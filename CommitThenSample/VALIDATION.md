# Validation record

Checked on 11 November 2026 using the bundled source and the recorded Python 3.12.12 dependencies.

- 54 selected tests passed: ECVRF vectors, protocol round trips, committed thresholds, Horvitz-Thompson estimation, and Merkle proofs.
- Rebuilt E1 and E2 revision-one summaries and the E3 base summaries from the included checkpoints.
- All 44,429 shared numeric fields in those three numeric-summary files matched their original values, excluding the E2 execution-time field.
- All capture pointers in the relocated inventory resolve to files inside this bundle.
- Repeated runtime-path preparation preserves valid local capture pointers.
- Computational source files match the original project byte-for-byte on the recorded platform.
- The complete source-data pipeline and all full-scale experiments were not rerun for this packaging task.
