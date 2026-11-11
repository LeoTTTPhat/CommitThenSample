#!/bin/sh
# E6 rev1 timing re-measurement on a quiet machine (19 Oct, after the external CI load ended). The first rev1 runs
# (P-core at 1-min load 65-200, E-core at 11-20) are kept in results/E6_rev1/loaded_runs/ with their summaries.
# Order = importance (pipeline, epochs, verifier, as-implemented, P flood); each run records its load; resumable.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
S=P1_CommitThenSample/scripts/15_e6.py
echo "$(date) start, load $(sysctl -n vm.loadavg)"
for r in 0 1 2 3 4; do .venv/bin/python -I -u $S pipeline-run --core P --rep $r --cached --rev rev1 || exit 1; done
for r in 0 1 2 3 4; do taskpolicy -b .venv/bin/python -I -u $S pipeline-run --core E --rep $r --cached --rev rev1 || exit 1; done
.venv/bin/python -I -u $S epochs-run --core P --rev rev1 || exit 1
taskpolicy -b .venv/bin/python -I -u $S epochs-run --core E --rev rev1 || exit 1
.venv/bin/python -I -u $S verifier-run --core P --cached --rev rev1 || exit 1
taskpolicy -b .venv/bin/python -I -u $S verifier-run --core E --cached --rev rev1 || exit 1
for r in 0 1 2 3 4; do .venv/bin/python -I -u $S pipeline-run --core P --rep $r --rev rev1 || exit 1; done
for r in 0 1 2 3 4; do taskpolicy -b .venv/bin/python -I -u $S pipeline-run --core E --rep $r --rev rev1 || exit 1; done
.venv/bin/python -I -u $S flood-run --core P || exit 1
.venv/bin/python -I -u $S summary --rev rev1 || exit 1
.venv/bin/python -I -u $S flood-summary || exit 1
.venv/bin/python -I -u P1_CommitThenSample/figures/tab5_e6_rev1.py || exit 1
echo "quiet re-measurement finished $(date), load $(sysctl -n vm.loadavg)" > P1_CommitThenSample/results/E6_rev1/QUIET_DONE
echo "$(date) done, load $(sysctl -n vm.loadavg)"
