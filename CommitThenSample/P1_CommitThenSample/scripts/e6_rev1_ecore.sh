#!/bin/sh
# E6 rev1, efficiency-core timings. Waits for a quiet machine (1-min load < 24 at two consecutive checks, i.e. the
# 24 performance cores absorb all other work and the 8 efficiency cores are free), then runs the E-core jobs
# (background QoS via taskpolicy -b = efficiency cores only), rebuilds the rev1 summaries and Table V, and writes
# results/E6_rev1/ECORE_DONE. Every stage is resumable (existing per-rep JSONs are skipped).
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
n=0; ok=0
while [ $n -lt 1500 ]; do
  l=$(sysctl -n vm.loadavg | awk '{print $2}')
  q=$(echo "$l < 24" | bc)
  if [ "$q" = "1" ]; then ok=$((ok+1)); else ok=0; fi
  if [ $ok -ge 2 ]; then break; fi
  n=$((n+1)); sleep 60
done
echo "$(date) starting E-core runs at load $l (waited $n min)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
S=P1_CommitThenSample/scripts/15_e6.py
for r in 0 1 2 3 4; do taskpolicy -b .venv/bin/python -I -u $S pipeline-run --core E --rep $r --cached --rev rev1 || exit 1; done
for r in 0 1 2 3 4; do taskpolicy -b .venv/bin/python -I -u $S pipeline-run --core E --rep $r --rev rev1 || exit 1; done
taskpolicy -b .venv/bin/python -I -u $S epochs-run --core E --rev rev1 || exit 1
taskpolicy -b .venv/bin/python -I -u $S verifier-run --core E --cached --rev rev1 || exit 1
taskpolicy -b .venv/bin/python -I -u $S flood-run --core E || exit 1
l2=$(sysctl -n vm.loadavg | awk '{print $2}')
.venv/bin/python -I -u $S summary --rev rev1 || exit 1
.venv/bin/python -I -u $S flood-summary || exit 1
.venv/bin/python -I -u P1_CommitThenSample/figures/tab5_e6_rev1.py || exit 1
echo "load at start $l, at end $l2, finished $(date)" > P1_CommitThenSample/results/E6_rev1/ECORE_DONE
echo "$(date) E-core runs done (load at end $l2)"
