#!/usr/bin/env bash
# Train + evaluate rlwp for several seeds, with logs and a results summary.
#
#   omp_tools/train_omp.sh [-t TASK] [-s "SEEDS"] [-n NAME] [--train-only | --eval-only] [--smoke] [--dry-run] [-- hydra overrides]
#
#   -t TASK       rlwp task config (cfg/task/<TASK>.yaml). Default: Lift_omp. Use Lift for the Panda baseline.
#   -s "SEEDS"    space-separated seeds. Default: "0 1 2"
#   -n NAME       run name prefix; each seed runs as <NAME>_s<seed>. Default: <TASK>_<date>-<time>
#   --train-only  skip evaluation
#   --eval-only   only evaluate existing <NAME>_s<seed> models
#   --smoke       tiny schedule (a few episodes) to check the pipeline end to end in ~1 min per seed
#   --dry-run     print the commands without running them
#   -- ...        extra hydra overrides passed to main.py, e.g. -- task.task.epoch_wp=100
#
# Outputs: models in rlwp/models/[<robot>/]<env>/<NAME>_s<seed>/, logs and summary.md in rlwp/logs/<NAME>/,
# TensorBoard runs in rlwp/runs/. See README_OMP.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLWP="$SCRIPT_DIR/../rlwp"
PYTHON="${PYTHON:-$SCRIPT_DIR/../../.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python3
export MUJOCO_GL="${MUJOCO_GL:-egl}"  # headless rendering backend; training itself does not render

TASK=Lift_omp
SEEDS="0 1 2"
NAME=""
DO_TRAIN=1
DO_EVAL=1
SMOKE=0
DRY=0
EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        -t) TASK="$2"; shift 2 ;;
        -s) SEEDS="$2"; shift 2 ;;
        -n) NAME="$2"; shift 2 ;;
        --train-only) DO_EVAL=0; shift ;;
        --eval-only) DO_TRAIN=0; shift ;;
        --smoke) SMOKE=1; shift ;;
        --dry-run) DRY=1; shift ;;
        --) shift; EXTRA=("$@"); break ;;
        -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done

[ -f "$RLWP/cfg/task/$TASK.yaml" ] || { echo "no task config $RLWP/cfg/task/$TASK.yaml" >&2; exit 2; }
[ -n "$NAME" ] || NAME="${TASK}_$(date +%Y%m%d-%H%M)"
if [ "$NAME" = "test" ] || [[ "$NAME" == test_* ]]; then
    echo "run name '$NAME' is reserved: the shipped Panda models live in models/<env>/test" >&2; exit 2
fi
if [ "$SMOKE" = 1 ]; then
    EXTRA=(task.task.epoch_wp=6 task.task.batch_size=3 task.task.exploration_epoch=2
           task.task.ensemble_sampling_epoch=3 task.task.averaging_noise_epoch=4 task.task.rand_reset_epoch=4
           task.task.num_eval=3 "${EXTRA[@]}")
fi

LOGS="$RLWP/logs/$NAME"
run() {  # run <log file> <main.py args...>
    local log="$1"; shift
    echo "+ (cd rlwp && $PYTHON main.py $*) > ${log#$RLWP/}"
    [ "$DRY" = 1 ] && return 0
    (cd "$RLWP" && "$PYTHON" main.py "$@") > "$log" 2>&1 || { echo "FAILED, see $log" >&2; tail -20 "$log" >&2; exit 1; }
}

[ "$DRY" = 1 ] || mkdir -p "$LOGS"
echo "task=$TASK seeds=($SEEDS) name=$NAME logs=${LOGS#$RLWP/}"
for s in $SEEDS; do
    if [ "$DO_TRAIN" = 1 ]; then
        start=$(date +%s)
        run "$LOGS/seed${s}_train.log" train=True task="$TASK" seed="$s" run_name="${NAME}_s$s" "${EXTRA[@]}"
        [ "$DRY" = 1 ] || echo "  seed $s trained in $(( $(date +%s) - start )) s"
    fi
    if [ "$DO_EVAL" = 1 ]; then
        # evaluation uses a different seed than training, so it sees new object placements
        run "$LOGS/seed${s}_eval.log" test=True task="$TASK" seed="$((1000 + s))" run_name="${NAME}_s$s" "${EXTRA[@]}"
        [ "$DRY" = 1 ] || grep -h "^EVAL" "$LOGS/seed${s}_eval.log" | sed 's/^/  /'
    fi
done

[ "$DRY" = 1 ] || "$PYTHON" "$SCRIPT_DIR/summarize_runs.py" "$NAME" --out "$LOGS/summary.md"
