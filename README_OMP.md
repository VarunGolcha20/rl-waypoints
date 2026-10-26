# rl-waypoints on OpenMANIPULATOR-P + OpenMANIPULATOR-X gripper

This fork adds a ROBOTIS **OpenMANIPULATOR-P** (6-DoF) with the **OpenMANIPULATOR-X gripper** to the bundled
robosuite, and lets `rlwp` train and evaluate on it. The original Panda workflow from [ReadMe.md](ReadMe.md) is
unchanged: Panda results are bit-identical with the same seed.

Background documents (in the parent folder `RLproj/`):
`PORTING_PLAN.md` (why), `RETRAINING_PLAN.md` (what changes for training, sim-to-real), `IMPLEMENTATION_LOG.md`
(every file changed, with verification results).

---

## 1. Setup

Everything runs from the project virtualenv (`RLproj/.venv`, CPU-only PyTorch, MuJoCo 3.1.6, robosuite installed
in editable mode from `robosuite/`):

```bash
cd ~/Desktop/RLproj/rl-waypoints
source ../.venv/bin/activate
export MUJOCO_GL=egl          # headless rendering for the check scripts; the training launcher sets this itself
```

## 2. Look at the robot

```bash
PYGLFW_LIBRARY_VARIANT=x11 python omp_tools/demo_omp_viewer.py           # scripted pick in Lift, loops
PYGLFW_LIBRARY_VARIANT=x11 python omp_tools/demo_omp_viewer.py --static  # robot at its start pose
```
MuJoCo viewer controls: left-drag rotate, right-drag pan, scroll zoom, `0` shows collision geoms. `PYGLFW_LIBRARY_VARIANT=x11`
avoids GLFW's native-Wayland backend.

## 3. Sanity checks (optional, rerun after changing the robot model)

| Script | Checks | Time |
|---|---|---|
| `python omp_tools/check_omp_arm.py` | arm MJCF alone: FK, self-collision, torque headroom, PD hold | ~10 s |
| `python omp_tools/check_omx_gripper.py` | gripper alone: opening range, grasp + lift of 24–50 mm cubes | ~30 s |
| `python omp_tools/check_omp_env.py [--env Lift]` | robot in a robosuite task: placement, OSC response, scripted pick | ~30 s |
| `python omp_tools/check_omp_workspace.py` | waypoint reachability grid → `workspace:` bounds for the task YAMLs | ~4.5 min |

Renders and raw results go to `omp_tools/out/` (git-ignored).

## 4. Training and evaluation

### 4.1 The launcher (recommended)

`omp_tools/train_omp.sh` trains and then evaluates each seed, saves logs, and writes a results summary.

```bash
# 1) check the pipeline end to end (tiny schedule, ~1 min per seed; results are meaningless)
omp_tools/train_omp.sh --smoke -s "0" -n smoke_check

# 2) full Lift on OM-P, 3 seeds (~25 min per seed on this CPU: ~20 min training + ~5 min eval)
omp_tools/train_omp.sh -t Lift_omp -s "0 1 2" -n omp_lift

# 3) Panda baseline with the same code and seeds (original Lift config)
omp_tools/train_omp.sh -t Lift -s "0 1 2" -n panda_lift

# 4) Stack on OM-P (~40 min per seed)
omp_tools/train_omp.sh -t Stack_omp -s "0 1 2" -n omp_stack
```

Options:

| Option | Meaning |
|---|---|
| `-t TASK` | task config `rlwp/cfg/task/<TASK>.yaml`. Default `Lift_omp` |
| `-s "SEEDS"` | seeds, run one after another. Default `"0 1 2"` |
| `-n NAME` | run name prefix. Seed *k* is saved as `<NAME>_s<k>`. Default `<TASK>_<date>-<time>`. `test` is refused (it is the shipped Panda models' name) |
| `--train-only` / `--eval-only` | skip one of the phases (`--eval-only` re-evaluates existing `<NAME>_s<k>` models) |
| `--smoke` | tiny schedule for pipeline checks |
| `--dry-run` | print the commands only |
| `-- key=value ...` | extra Hydra overrides for `main.py`, e.g. `-- task.task.epoch_wp=100 task.task.num_eval=50` |

Long runs keep going after you log out if you start them with `nohup`:
```bash
nohup omp_tools/train_omp.sh -t Lift_omp -n omp_lift > omp_lift.out 2>&1 &
```
Seeds run sequentially. To use more CPU cores, start one launcher per seed with the **same** `-n`
(e.g. `-s 0`, `-s 1`, `-s 2` in three terminals). Each one prints a summary of whatever runs exist at the time it finishes.

### 4.2 Where things go

| What | Where |
|---|---|
| Reward-model ensembles | `rlwp/models/OpenManipulatorP/<Env>/<NAME>_s<k>/model_<wp>_<i>` (Panda: `rlwp/models/<Env>/<NAME>_s<k>/`, the original layout) |
| Training curve data | `.../<NAME>_s<k>/data.pkl` (`episode`, `reward`, `success` per training episode) |
| Evaluation data | `.../<NAME>_s<k>/eval_data.pkl` (`reward`, `success` per evaluation episode) |
| Logs | `rlwp/logs/<NAME>/seed<k>_{train,eval}.log` |
| Results table | `rlwp/logs/<NAME>/summary.md` (also printed at the end) |
| TensorBoard | `rlwp/runs/ours_<robot>_<NAME>_s<k><HH-MM>/`. View with `tensorboard --logdir rlwp/runs` (reward, success, critic loss) |

### 4.3 Monitoring and results

```bash
tail -f rlwp/logs/omp_lift/seed0_train.log         # one line per episode: waypoint, reward, success
tensorboard --logdir rlwp/runs                     # curves in the browser
python omp_tools/summarize_runs.py omp_lift        # table across seeds at any time
```
The summary reports per seed: training episodes, reward and success rate over the last 50 training episodes,
and evaluation reward (mean ± std) and success rate. If there are several seeds, it adds the mean across seeds.

### 4.4 Running `main.py` directly

The launcher only wraps the original entry point. Run from `rlwp/`:
```bash
cd rlwp
python main.py train=True task=Lift_omp seed=0 run_name=my_run          # train
python main.py test=True  task=Lift_omp seed=1000 run_name=my_run       # evaluate (prints an EVAL line)
python main.py test=True  task=Lift run_name=test                       # shipped Panda model
```
`train=True test=True` in one call does **not** run the evaluation, because `train` ends with `exit()` (original
behaviour). Run the two separately.

## 5. Configuration reference

New keys. All are optional, and leaving them out gives the original behaviour:

| Key | Where | Meaning |
|---|---|---|
| `seed` | `cfg/config.yaml` / command line | seeds python, numpy (incl. robosuite object placement) and torch. `null` = unseeded |
| `env.controller` | task YAML | robosuite controller config name (`robosuite/robosuite/controllers/config/<name>.json`). Unset = generic `OSC_POSE` |
| `env.workspace.low/high` | task YAML | absolute world-frame bounds (m) for the waypoint positions. Used by exploration, the SLSQP constraint and the injected noise. Unset = ±`action_space` around the start pose |
| `env.orientation_hold` | task YAML | for position-only tasks, actively keeps the gripper at its start orientation (pointing down) |

OM-P task configs: `cfg/task/Lift_omp.yaml`, `cfg/task/Stack_omp.yaml`. They use the paper's schedules (so the
curves compare directly with Panda), the `default_open_manipulator_p` controller (coupled OSC), orientation hold,
and the workspace boxes from `check_omp_workspace.py`:

| Task | workspace low | workspace high |
|---|---|---|
| Lift_omp | [-0.20, -0.25, 0.83] | [0.05, 0.25, 0.98] |
| Stack_omp | [-0.20, -0.25, 0.83] | [0.10, 0.25, 0.93] |

If you change the robot model, its base placement or the controller, rerun `omp_tools/check_omp_workspace.py` and paste
the printed `workspace:` blocks into the task YAMLs.

Door, NutAssembly and PickPlace don't have OM-P configs yet. Their layouts need adapting first (see `RETRAINING_PLAN.md` §7).

## 6. Robot model summary

* **Arm**: `robosuite/robosuite/models/assets/robots/open_manipulator_p/robot.xml` (from ROBOTIS' URDF, real
  inertias, hardware joint ranges, Dynamixel-P torque limits). Class `OpenManipulatorP`.
* **Gripper**: `robosuite/robosuite/models/assets/grippers/omx_gripper.xml` (from ROBOTIS' OM-X MuJoCo model,
  20–80 mm opening, box fingerpads), class `OMXGripper`, on a 10 mm adapter plate.
* **Mount**: the robot stands on a **10 cm riser on the table top** (`OMPRiserMount`), base at x = −0.32 m. With the
  gripper pointing down, OM-P's wrist (J5, ±90°) limits the grip point to ~0.10 m above the base, hence the riser.
  Mirror this on the real robot.
* **Start pose**: gripper pointing down, 0.17 m above the table, 0.22 m in front of the base.
* **Controller**: `controllers/config/default_open_manipulator_p.json` = robosuite OSC_POSE with
  `uncouple_pos_ori: false`. The default uncoupled OSC moves this 6-DoF arm at half speed along x.

## 7. Troubleshooting

* `AttributeError: module 'robosuite' has no attribute 'make'`: you started Python *interactively or with `-c`* in
  `rl-waypoints/`, where the folder `robosuite/` shadows the package. Run from `rlwp/` or anywhere else (scripts run as
  files are fine).
* Viewer window problems on Wayland: prefix with `PYGLFW_LIBRARY_VARIANT=x11`.
* Loading the shipped Panda models on a CPU-only machine needs `map_location`, which is now built in.
* `run name 'test' is reserved`: choose another `-n`. The shipped Panda models are stored under `test`.
