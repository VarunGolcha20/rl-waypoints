#!/usr/bin/env bash
# Train + evaluate the object-relative Stack model (PLAN_RELATIVE_WAYPOINTS.md) in one go, detached from the terminal:
# 3 seeds trained in parallel, evaluated with 20 optimiser restarts, summarised, then success maps of the best seed and
# of omx_stack_v4_s1 on the same cubeA grid (incl. the left side) to compare generalization.
#
#   cd ~/Documents/RL/rl-waypoints
#   setsid nohup omp_tools/train_rel_detached.sh > rlwp/logs/omx_stack_rel.out 2>&1 < /dev/null &
#
#   progress:   tail -f rlwp/logs/omx_stack_rel.out
#               tail -f rlwp/logs/omx_stack_rel/seed0_train.log       (one line per episode)
#   stop:       pkill -f 'train_rel_detache[d]'; pkill -f 'run_name=omx_stack_rel_s'
#
# Options (environment): NAME (default omx_stack_rel), TASK (Stack_omx_rel), SEEDS ("0 1 2"), BLOCK ("5.0 17.5": the
# landing block's position for the maps), THREADS (4 torch threads per run), SMOKE=1 (tiny schedule, pipeline check).
# Results: rlwp/logs/$NAME/summary.md, rlwp/logs/$NAME/map_*.txt, and the end of this script's output.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$SCRIPT_DIR/.."
PYTHON="${PYTHON:-$REPO/../.venv/bin/python}"
NAME="${NAME:-omx_stack_rel}"
TASK="${TASK:-Stack_omx_rel}"
SEEDS="${SEEDS:-0 1 2}"
BLOCK="${BLOCK:-5.0 17.5}"
export OMP_NUM_THREADS="${THREADS:-4}" MKL_NUM_THREADS="${THREADS:-4}" MUJOCO_GL=egl
SMOKE_ARG=()
MAP_GRID=(--x 10 20 --y -10 12 --step 2)
if [ "${SMOKE:-0}" = 1 ]; then SMOKE_ARG=(--smoke); MAP_GRID=(--x 12 16 --y -2 8 --step 4); fi
LOGS="$REPO/rlwp/logs/$NAME"
mkdir -p "$LOGS"
cd "$REPO"
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(stamp)] $NAME: task $TASK, seeds ($SEEDS), $OMP_NUM_THREADS threads per run, host $(hostname), commit $(git rev-parse --short HEAD)"

# 1) training, all seeds in parallel (each run: rlwp/logs/$NAME/seed<k>_train.log)
t0=$(date +%s)
pids=()
for s in $SEEDS; do
    omp_tools/train_omp.sh -t "$TASK" -s "$s" -n "$NAME" --train-only "${SMOKE_ARG[@]}" > "$LOGS/launcher_train_s$s.out" 2>&1 &
    pids+=($!)
    echo "[$(stamp)] training seed $s started (pid $!)"
done
failed=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then echo "[$(stamp)] training of a seed FAILED, see $LOGS/launcher_train_s*.out"; failed=1; fi
done
echo "[$(stamp)] training done in $(( ($(date +%s) - t0) / 60 )) min"
[ "$failed" = 0 ] || { tail -20 "$LOGS"/launcher_train_s*.out; exit 1; }

# 2) evaluation with 20 optimiser restarts (5 underrate Stack models, OMX_PORT.md P8), seeds in parallel
pids=()
for s in $SEEDS; do
    omp_tools/train_omp.sh -t "$TASK" -s "$s" -n "$NAME" --eval-only "${SMOKE_ARG[@]}" -- n_inits=20 > "$LOGS/launcher_eval_s$s.out" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || echo "[$(stamp)] an evaluation FAILED, see $LOGS/launcher_eval_s*.out"; done
echo "[$(stamp)] evaluation done"
"$PYTHON" omp_tools/summarize_runs.py "$NAME" --out "$LOGS/summary.md" > /dev/null 2>&1 || true

# 3) best seed by evaluation success, then success maps: the relative model vs omx_stack_v4_s1, same cubeA grid
BEST=$("$PYTHON" - "$NAME" "$SEEDS" <<'EOF'
import pickle, sys, os
name, seeds = sys.argv[1], sys.argv[2].split()
best, rate = None, -1.0
for s in seeds:
    f = "rlwp/models/OpenManipulatorX/Stack/%s_s%s/eval_data.pkl" % (name, s)
    if os.path.exists(f):
        d = pickle.load(open(f, "rb")); r = sum(d["success"]) / max(1, len(d["success"]))
        if r > rate: best, rate = s, r
print("%s_s%s" % (name, best) if best is not None else "")
EOF
)
echo "[$(stamp)] best seed: ${BEST:-none}"
if [ -n "$BEST" ]; then
    "$PYTHON" omx_hw/sim_sweep.py --task Stack --config "$TASK" --run "$BEST" --object cubeA --other $BLOCK "${MAP_GRID[@]}" \
        --json "$LOGS/map_relative.json" > "$LOGS/map_relative.txt" 2>&1 &
    p1=$!
    "$PYTHON" omx_hw/sim_sweep.py --task Stack --run omx_stack_v4_s1 --object cubeA --other $BLOCK "${MAP_GRID[@]}" \
        --json "$LOGS/map_v4.json" > "$LOGS/map_v4.txt" 2>&1 &
    p2=$!
    wait $p1 $p2
    echo "[$(stamp)] maps done"
fi

# 4) final report (also at the end of this output)
filter() { grep -vE 'robosuite WARNING|Gym has been|Please upgrade|Users of this|See the migration'; }
echo
echo "================================================================ FINISHED $(stamp), $(( ($(date +%s) - t0) / 60 )) min"
cat "$LOGS/summary.md" 2>/dev/null
echo
grep -h '^EVAL' "$LOGS"/seed*_eval.log 2>/dev/null
for m in relative v4; do
    echo
    echo "---- success map, cubeA grid, landing block at ($BLOCK) cm: $m model"
    filter < "$LOGS/map_$m.txt" 2>/dev/null | tail -n +2
done
echo
echo "Models: rlwp/models/OpenManipulatorX/Stack/${NAME}_s*/   Logs: rlwp/logs/$NAME/"
