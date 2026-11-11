#!/bin/sh
# E6 rev1: commit time on the real scan-heavy LP-IoT23 streams, reviewed vs rev1 rolling statistics, P then E core.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
S=P1_CommitThenSample/scripts/15b_e6_realstream.py
echo "$(date) start, load $(sysctl -n vm.loadavg)"
for s in lp_iot23_49-1_p10000 lp_iot23_48-1_p10000 lp_iot23_7-1_p10000; do
  for i in incremental resort; do .venv/bin/python -I -u $S run --core P --stream $s --impl $i || exit 1; done
done
for s in lp_iot23_49-1_p10000 lp_iot23_48-1_p10000 lp_iot23_7-1_p10000; do
  for i in incremental resort; do taskpolicy -b .venv/bin/python -I -u $S run --core E --stream $s --impl $i || exit 1; done
done
.venv/bin/python -I -u $S summary > /dev/null || exit 1
echo "$(date) done, load $(sysctl -n vm.loadavg)"
