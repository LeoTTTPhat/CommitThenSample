# Commit-then-Sample reproduction artifact

This folder includes the implementation, required traffic datasets, experiment configurations, trained utility models, prepared streams, retained evidence, and recorded experiment outputs for Commit-then-Sample.


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
| `P1_CommitThenSample/src/` | Computational implementation |
| `P1_CommitThenSample/scripts/` | Data preparation and experiment entry points |
| `P1_CommitThenSample/config/` | Experiment settings and random seeds |
| `P1_CommitThenSample/tests/` | Existing correctness and regression checks |
| `P1_CommitThenSample/results/` | Prepared inputs, trained checkpoints, cached scores, and recorded outputs |
| `P1_CommitThenSample/figures/` | Plotting programs and reference images |
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
.venv/bin/python reproduce.py run -- P1_CommitThenSample/scripts/SCRIPT.py --help
```

The launcher runs heavy jobs under the included memory supervisor. It defaults to two workers, limits thread pools, and applies the recorded memory budget. Set `MEMGUARD_WORKERS=1` to reduce parallelism. Large-data regeneration can require substantially more memory and time than summary rebuilding. Hardware timing results depend on the machine and its background load.

Scripts normally resume from completed checkpoints. Consult each script's `--help` for its available `--force` or `--overwrite` option before requesting fresh computation. Retain a backup of the supplied `results/` directory when recomputing. Reference summaries provide the expected outputs; numerical tolerances may be necessary across numerical-library or hardware changes.


## Native tools

For raw capture processing, install Suricata 8.0.7 and the Wireshark command-line tools `capinfos`, `editcap`, and `tshark`. On macOS, `brew install suricata wireshark` provides these programs; compare the installed Suricata version with 8.0.7 before attempting an exact replay. The recorded detection rules and configuration are included in `data/rules/`. Beacon chain information and historical rounds are included in `data/beacon/`; historical replay does not need a live beacon request.

## Rebuild summaries and run experiments

Rebuild the E1/E2 revision-one summaries and the E3 base summaries from the included experiment records:

```sh
.venv/bin/python reproduce.py summaries
```

Run the primary revision-one retention and validity experiments:

```sh
.venv/bin/python reproduce.py run -- P1_CommitThenSample/scripts/10_e1.py --config P1_CommitThenSample/config/experiments/E1_rev1.yaml --exp-root E1_rev1 --workers 2
.venv/bin/python reproduce.py run -- P1_CommitThenSample/scripts/11_e2.py --exp-root E2_rev1 --workers 2
.venv/bin/python reproduce.py run -- P1_CommitThenSample/scripts/12_e3.py --workers 2
.venv/bin/python reproduce.py run -- P1_CommitThenSample/scripts/13b_e4_rev1.py
```

For the two stream-recipe sensitivity variants, `zq` and `z0`, use the supplied variant settings and the `--variant` argument of the stream-building, utility-training, design-building, and retention scripts. The variant streams, models, and existing outputs are already included.

| Experiment | Entry point |
|---|---|
| E1 retention | `scripts/10_e1.py` |
| E2 validity and fixed-time comparisons | `scripts/11_e2.py` |
| E3 model and policy analysis | `scripts/12_e3.py` |
| E4 incident retrieval | `scripts/13b_e4_rev1.py` |
| E5 security trials | `scripts/14_e5.py` |
| E6 overheads and flood measurements | `scripts/15_e6.py`, `scripts/15b_e6_realstream.py` |
| E7 design tradeoffs | `scripts/16_e7.py` |
| E8 case study | `scripts/17_e8.py` |

Paths in this table are relative to `P1_CommitThenSample/`. The security, overhead, and case-study scripts accept a stage argument; inspect their help before choosing a stage. E5 and E8 default to revision one, while E6 needs `--rev rev1`.

## Regenerate prepared data

The raw traffic inputs are in `data/iot23/`, `data/ciciot2023/`, `data/edgeiiot/`, and `data/gotham2025/`. The prepared tables, stream recipes, model checkpoints, and experiment configurations are included for direct replay.

The dependency order for a complete rebuild is:

1. `scripts/00_inventory.py`: inspect captures and source labels.
2. `scripts/01_extract_units.py --dataset DATASET`: extract units and replay captures.
3. `scripts/02a_build_rules.py`: prepare the recorded content-rule set.
4. `scripts/02_run_suricata.py --dataset DATASET`: compute content alerts.
5. `scripts/03_build_tables.py --dataset DATASET`: build labelled unit tables.
6. `scripts/03b_build_streams.py --workers 2`: construct experiment streams.
7. `scripts/04_train_utility.py --workers 2`: fit utility models and calibration.
8. `scripts/05_designs.py --workers 2`: construct the sampling designs.
9. Run the experiment entry points above, then rebuild summaries.

`DATASET` is one of `iot23`, `ciciot2023_benign`, `edgeiiot`, or `gotham`. Configuration values and recorded source hashes determine run identities. Runtime relocation preserves the computational source on the recorded platform; executable lookup adjustments on another installation can create new run identities.
