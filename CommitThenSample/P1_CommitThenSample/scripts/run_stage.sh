#!/bin/sh
# Canonical invocations of the P1 data-pipeline stages (DESIGN §8.1), run from anywhere.
# Usage: P1_CommitThenSample/scripts/run_stage.sh <stage> [dataset] [extra args...]
#   stages: s0 | s1 <ds> | s1_edge | s2a | s2 <ds> | s2_orig <ds> | s2_audit <ds> | s3 <ds> | s3_labels <ds> | tests
#   datasets: iot23 | ciciot2023_benign | edgeiiot | gotham
# Every heavy job runs under memguard (--workers sets MEMGUARD_WORKERS; -u is required with -I).
set -eu
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
G=".venv/bin/python memguard/memguard.py run"
PY=".venv/bin/python -I -u P1_CommitThenSample/scripts"
stage="${1:-}"; shift || true
case "$stage" in
  s0)        $G --name p1_s0 --workers 16 -- $PY/00_inventory.py "$@" ;;
  s1)        ds="$1"; shift; $G --name "p1_s1_$ds" --workers 14 -- $PY/01_extract_units.py --dataset "$ds" "$@" ;;
  s1_edge)   $G --name p1_s1_edge --workers 12 -- $PY/01_extract_units.py --dataset edgeiiot --no-floods "$@"
             $G --name p1_s1_edge_floods --workers 4 -- $PY/01_extract_units.py --dataset edgeiiot --floods-only "$@" ;;
  s2a)       $G --name p1_s2a --workers 1 -- $PY/02a_build_rules.py "$@" ;;
  s2)        ds="$1"; shift; $G --name "p1_s2_$ds" --workers 4 -- $PY/02_run_suricata.py --dataset "$ds" "$@" ;;
  s2_orig)   ds="$1"; shift; $G --name "p1_s2orig_$ds" --workers 4 -- $PY/02_run_suricata.py --dataset "$ds" --runs orig "$@" ;;
  s2_audit)  ds="$1"; shift; $G --name "p1_s2audit_$ds" --workers 1 -- $PY/02_run_suricata.py --dataset "$ds" --audit-only ;;
  s3)        ds="$1"; shift; $G --name "p1_s3_$ds" --workers 6 -- $PY/03_build_tables.py --dataset "$ds" "$@" ;;
  s3_labels) ds="$1"; shift; $G --name "p1_s3labels_$ds" --workers 6 -- $PY/03_build_tables.py --dataset "$ds" --labels-only "$@" ;;
  tests)     cd P1_CommitThenSample && ../.venv/bin/python -I -m pytest -q tests/test_flowtable.py tests/test_detectors.py \
               tests/test_replay_vlan.py tests/test_encode.py tests/test_labels_iot23.py "$@" ;;
  *) echo "usage: $0 {s0|s1 <ds>|s1_edge|s2a|s2 <ds>|s2_orig <ds>|s2_audit <ds>|s3 <ds>|s3_labels <ds>|tests}" >&2; exit 2 ;;
esac
