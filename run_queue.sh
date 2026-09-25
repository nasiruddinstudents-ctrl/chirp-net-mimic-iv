#!/usr/bin/env bash
# Resumable queue for run_unified_nodx.py. Skips any run whose results file exists.
# Usage (on the Vast box, from /workspace):
#   SMOKE=1 JOBS=3 bash run_queue.sh          # quick pipeline check, all conditions
#   JOBS=3 nohup bash run_queue.sh > queue.log 2>&1 &   # the real runs
cd /workspace
JOBS=${JOBS:-2}
SMOKE=${SMOKE:-0}
SFX=""; [ "$SMOKE" = "1" ] && SFX="_smoke"
mkdir -p logs_unified$SFX predictions_unified$SFX results_unified$SFX

EDGE_NORM=${EDGE_NORM:-zlog}
# Legacy-protocol full-model predictions are only reusable when EDGE_NORM=clamp.
if [ "$SMOKE" != "1" ] && [ "$EDGE_NORM" = "clamp" ]; then
  for s in 42 43 44 45 46; do
    src=predictions_nodx/chirp_nodx_seed${s}_test_predictions.csv
    [ -f "$src" ] && cp -n "$src" predictions_unified/full_seed${s}_test_predictions.csv \
      && [ ! -f results_unified/full_seed${s}.csv ] && echo "condition,seed,edge_norm" > results_unified/full_seed${s}.csv \
      && echo "full,$s,clamp_reused" >> results_unified/full_seed${s}.csv
  done
fi

# Priority order: reviewer-required first, then protocol-matched reruns of Tables 5/6.
CONDS=${CONDS:-"full no_aux homogeneous_small heterogeneous_small homogeneous_large vitlab_only edge_value_zeroed edge_time_zeroed edge_time_permuted no_reverse"}
SEEDS="42 43 44 45 46"; [ "$SMOKE" = "1" ] && SEEDS="42"

for c in $CONDS; do for s in $SEEDS; do
  [ -f results_unified$SFX/${c}_seed${s}.csv ] || echo "$c $s"
done; done | xargs -P "$JOBS" -L 1 bash -c \
  'CONDITION=$0 SEED=$1 SMOKE='"$SMOKE"' EDGE_NORM='"$EDGE_NORM"' python3 -u run_unified_nodx.py > logs_unified'"$SFX"'/${0}_seed${1}.log 2>&1; echo "finished $0 $1 (exit $?)"'

echo "=== queue done ==="
python3 - <<PY
import pandas as pd, glob
fs = sorted(glob.glob("results_unified$SFX/*.csv"))
if fs:
    df = pd.concat(pd.read_csv(f) for f in fs)
    print(df.groupby("condition")[["test_auroc","test_auprc","params","best_epoch","skipped_batches","minutes"]]
            .agg(["mean","std","count"]).round(4).to_string())
PY
