# Validation record

Checked on 11 November 2026 using the bundled source and the recorded Python 3.12.12 dependencies.

- 53 selected CPU tests passed: CRC, cascade logic, ChainForge generation, feature leakage, and evidence-card rendering.
- One MLX integration test was deselected in the CPU smoke action. A direct attempt reached the runtime import and failed because this sandbox has no accessible Metal device; GPU execution remains unverified here.
- Rebuilt E1 and E7 numeric summaries from the included experiment checkpoints.
- All 72,768 shared numeric fields in those two files matched their original values.
- The frozen evaluation registry and prompt-freeze JSON match the originals byte-for-byte.
- The three configured MLX model snapshots are included as regular local files.
- Computational source files match the original project byte-for-byte.
- The complete source-data pipeline and all full-scale experiments were not rerun for this packaging task.
