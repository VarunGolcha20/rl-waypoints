"""
Summarises rlwp runs launched by train_omp.sh: per seed and across seeds, from the data.pkl (training) and
eval_data.pkl (evaluation) files under rlwp/models/.

    python omp_tools/summarize_runs.py <run name prefix> [--out summary.md]

Finds every rlwp/models/**/<prefix>_s<seed>/ directory, so it works for any robot / task layout.
"""
import argparse
import glob
import os
import pickle
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "..", "rlwp", "models")


def load(path):
    return pickle.load(open(path, "rb")) if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="run name prefix (train_omp.sh -n), seeds are <name>_s<seed>")
    ap.add_argument("--models", default=MODELS)
    ap.add_argument("--last", type=int, default=50, help="training episodes used for the 'late training' numbers")
    ap.add_argument("--out", help="also write the table to this markdown file")
    args = ap.parse_args()

    dirs = sorted(d for d in glob.glob(os.path.join(args.models, "**", args.name + "_s*"), recursive=True)
                  if re.search(re.escape(args.name) + r"_s\d+$", d))
    if not dirs:
        raise SystemExit("no runs found for '%s' under %s" % (args.name, os.path.abspath(args.models)))

    lines = ["| run | train episodes | late-train reward | late-train success | eval episodes | eval reward | eval success |",
             "|---|---|---|---|---|---|---|"]
    eval_rewards, eval_success = [], []
    for d in dirs:
        tr, ev = load(os.path.join(d, "data.pkl")), load(os.path.join(d, "eval_data.pkl"))
        row = [os.path.relpath(d, args.models)]
        if tr:
            r = np.array(tr["reward"][-args.last:])
            s = np.array(tr.get("success", [np.nan] * len(tr["reward"]))[-args.last:], dtype=float)
            row += [str(len(tr["reward"])), "%.2f" % r.mean(), "%.0f%%" % (100 * np.nanmean(s)) if len(s) else "-"]
        else:
            row += ["-", "-", "-"]
        if ev:
            r = np.array(ev["reward"])
            s = np.array(ev.get("success", [np.nan] * len(r)), dtype=float)
            row += [str(len(r)), "%.2f ± %.2f" % (r.mean(), r.std()), "%.0f%%" % (100 * np.nanmean(s))]
            eval_rewards.append(r.mean())
            eval_success.append(np.nanmean(s))
        else:
            row += ["-", "-", "-"]
        lines.append("| " + " | ".join(row) + " |")
    if len(eval_rewards) > 1:
        lines.append("| **mean over %d seeds** | | | | | %.2f ± %.2f | %.0f%% ± %.0f%% |" % (
            len(eval_rewards), np.mean(eval_rewards), np.std(eval_rewards),
            100 * np.mean(eval_success), 100 * np.std(eval_success)))
    table = "\n".join(lines)
    print(table)
    if args.out:
        with open(args.out, "w") as f:
            f.write("# Results: %s\n\n%s\n\n(late-train = last %d training episodes, i.e. the final waypoint's bandit)\n" % (
                args.name, table, args.last))
        print("\nwritten to", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
