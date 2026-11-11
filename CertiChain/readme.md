# CertiChain reproduction artifact

This folder includes the implementation, required traffic datasets, ChainForge incidents, statistical models, cached language-model scores, loss banks, frozen evaluation settings, and recorded experiment outputs for CertiChain.


## Setup and quick validation

Use Python 3.12 on macOS with Apple silicon. The original environment used Python 3.12.12 on an Apple M3 Ultra with 512 GB unified memory. Exact Python package versions are recorded in `requirements.lock.txt`.

Run the following commands inside this `Artifact` directory:

```sh
sh bootstrap.sh
.venv/bin/python reproduce.py check
.venv/bin/python reproduce.py smoke
```

Set `PYTHON=/path/to/python3.12` when the interpreter has a different name. Setup installs the pinned dependencies and this local package, then relocates runtime file pointers. Package installation needs access to a Python package index. The bundled inputs and checkpoints support the later workflows locally.

The `check` action checks file presence, sizes, control-file checksums, and Python import locations. For a complete SHA-256 inventory of the larger files, use `.venv/bin/python reproduce.py check --hash-all`; this reads the complete data collection and writes `full_sha256.json`.

## Contents

| Path | Purpose |
|---|---|
| `P2_CertiChain/src/` | Computational implementation |
| `P2_CertiChain/scripts/` | Data preparation and experiment entry points |
| `P2_CertiChain/configs/` | Experiment settings and random seeds |
| `P2_CertiChain/tests/` | Existing correctness and regression checks |
| `P2_CertiChain/results/` | Prepared inputs, trained checkpoints, cached scores, and recorded outputs |
| `P2_CertiChain/figures/` | Plotting programs and reference images |
| `data/` | Required source datasets and supporting resources |
| `memguard/` | Process supervision and memory controls |
| `tools/` | Shared data and runtime utilities |
| `environment.json` | Recorded environment and resource details |
| `files.jsonl` | Complete file inventory with sizes |
| `control_sha256.json` | SHA-256 hashes for code, configurations, and reference summaries |
| `VALIDATION.md` | Checks performed on this copy |
| `reference_summaries/` | Unchanged numeric summaries for output comparison |

The nested project directory preserves the original relative-path layout. All data and checkpoints are regular files within this folder. The copies use copy-on-write storage where supported, so edits remain independent. Source attribution and acquisition details are retained in each dataset's `MANIFEST.md`. When moving this folder to another path, run `.venv/bin/python prepare_runtime.py` before executing the workflows.

## Running individual stages

Use the launcher to run a bundled script from this directory:

```sh
.venv/bin/python reproduce.py run -- P2_CertiChain/scripts/SCRIPT.py --help
```

The launcher runs heavy jobs under the included memory supervisor. It defaults to two workers, limits thread pools, and applies the recorded memory budget. Set `MEMGUARD_WORKERS=1` to reduce parallelism. Large-data regeneration can require substantially more memory and time than summary rebuilding. Hardware timing results depend on the machine and its background load.

Scripts normally resume from completed checkpoints. Consult each script's `--help` for its available `--force` or `--overwrite` option before requesting fresh computation. Retain a backup of the supplied `results/` directory when recomputing. Reference summaries provide the expected outputs; numerical tolerances may be necessary across numerical-library or hardware changes.


The smoke action runs CPU checks; the MLX integration check remains in the test suite and requires an accessible Metal device.

## Native tools and model runtime

`tshark` is needed to rebuild the Gotham payload reports. The language-model runtime uses MLX on macOS with Apple silicon. The three recorded model snapshots are bundled in `data/huggingface/hub/`; their identifiers and exact revisions are recorded in `environment.json`. The launcher sets `HF_HUB_CACHE` to that local cache. Statistical replay from the supplied scores and loss banks does not need a model download or fresh language-model inference.

`results/PREREG_2026-10-21.json` and the frozen prompt resources under `results/gates/` are included byte-for-byte. Evaluation scripts validate the frozen registry hash. Keep these files intact when rerunning evaluations or regenerating downstream inputs.

## Rebuild summaries and run experiments

Rebuild the E1 and E7 numeric summaries from the included experiment cells:

```sh
.venv/bin/python reproduce.py summaries
```

Rerun the configured validity, power, shift, language-model, chain, and case-study evaluations:

```sh
.venv/bin/python reproduce.py run -- P2_CertiChain/scripts/12_run_experiment.py --exp E1,E7,E15,E2,E11,E5,E3,E4,validity --workers 2
```

Existing cells are resumed. To recompute an evaluation, add `--overwrite`; `--no-boot` omits bootstrap intervals and therefore produces a reduced evaluation.

The `cf_gotham_v2` and `cf_gotham_t_v2` inputs, `S_GOT2` and `S_CF2` scorers, and revision analyses are included alongside the earlier variants. Additional registered datasets can be selected with the experiment driver's `--register`, `--datasets`, `--ds-map`, and `--tag` options. Revision analysis entry points are available through:

```sh
.venv/bin/python reproduce.py run -- P2_CertiChain/scripts/21_revision_x2.py --help
```

## Regenerate prepared data and scores

| Stage | Entry point and example selection |
|---|---|
| IoT-23 ingestion | `scripts/01_ingest_iot23.py` |
| TON ingestion | `scripts/02_ingest_ton.py` |
| Gotham ingestion | `scripts/03_ingest_gotham.py` |
| Gotham payload reports | `scripts/03b_gotham_payload.py` |
| Episode libraries | `scripts/04_build_libraries.py --variant ton` and `--variant gotham` |
| ChainForge generation | `scripts/05_make_chainforge.py --variant cf_ton` |
| Real incidents | `scripts/06_make_real_incidents.py` |
| Units, candidates, ground truth | `scripts/07_units_candidates_gt.py --dataset cf_ton --workers 2` |
| Statistical scoring | `scripts/08_fit_stats.py --scorer S_TON --score` |
| Language-model scoring | `scripts/09_score_llm.py --model main --must` |
| Fusion models | `scripts/10_fit_fusion.py --scorer S_TON --score` |
| Loss banks | `scripts/11_precompute_losses.py --dataset cf_ton --scorers S_TON.stat --n-jobs 2` |
| Baselines | `scripts/16_baselines.py --dataset cf_ton` |
| Evaluation | `scripts/12_run_experiment.py --exp E1 --workers 2` |

Paths in the table are relative to `P2_CertiChain/`. Run stages in this order when rebuilding from source data, and repeat the dataset-dependent stages for every variant selected by the evaluation. Source inputs are included under `data/iot23/`, `data/ton_iot/`, `data/gotham2025/`, and `data/attack/`. The model snapshot cache is included separately under `data/huggingface/`.

The experiment records include aggregated per-split data and bootstrap outputs. Redundant individual split-cache files are omitted wherever the same directory has the complete `per_split.parquet` checkpoint; they can be regenerated by the evaluation driver. The earlier variants and all recorded seeds remain available for side-by-side reproduction.
