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
PYGLFW_LIBRARY_VARIANT=x11 python omp_tools/show_omp_tasks.py             # every task's environment, one window each
PYGLFW_LIBRARY_VARIANT=x11 python omp_tools/show_omp_tasks.py --env Stack # one task
MUJOCO_GL=egl python omp_tools/show_omp_tasks.py --snapshot               # PNG per task in omp_tools/out/
```
`show_omp_tasks.py` shows the environments only, with the adapted `_omp` layouts (it reads `env_kwargs` from the
task YAMLs): the robot holds its start pose while the physics runs (no learned or
scripted motion). Close a window to move on to the next task. Backspace re-randomises the objects, and Ctrl + right-drag pushes them.
MuJoCo viewer controls: left-drag rotate, right-drag pan, scroll zoom, `0` shows collision geoms. `PYGLFW_LIBRARY_VARIANT=x11`
avoids GLFW's native-Wayland backend.

## 3. Sanity checks (optional, rerun after changing the robot model)

| Script | Checks | Time |
|---|---|---|
| `python omp_tools/check_omp_arm.py` | arm MJCF alone: FK, self-collision, torque headroom, PD hold | ~10 s |
| `python omp_tools/check_omx_gripper.py` | gripper alone: opening range, grasp + lift of 24–50 mm cubes | ~30 s |
| `python omp_tools/check_omp_env.py [--env Lift]` | robot in a robosuite task: placement, OSC response, scripted pick | ~30 s |
| `python omp_tools/check_omp_tasks.py [--tasks ...]` | every `_omp` task solvable with 2 waypoints inside its workspace box (scripted waypoints through rlwp's action rule) | ~3 min |
| `python omp_tools/check_omp_workspace.py [--reuse]` | waypoint reachability grid → `workspace:` bounds for the task YAMLs (`--reuse` recomputes the boxes from the saved grid) | ~7 min (`--reuse`: seconds) |

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

# 4) the other adapted tasks. Measured rate ~5.3 s/episode with 3 seeds in parallel: Stack / Door ~55 min,
#    NutAssembly ~60 min, PickPlace ~90 min (Lift took ~35 min)
omp_tools/train_omp.sh -t Stack_omp -s "0 1 2" -n omp_stack
omp_tools/train_omp.sh -t NutAssembly_omp -s "0 1 2" -n omp_nut
omp_tools/train_omp.sh -t PickPlace_omp -s "0 1 2" -n omp_pickplace     # object=can is added automatically
omp_tools/train_omp.sh -t Door_omp -s "0 1 2" -n omp_door
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

### 4.4 Trained models and watching them

Trained OM-P models live in `rlwp/models/OpenManipulatorP/<Env>/<run>/` (reward-model ensembles `model_<waypoint>_<i>`,
`data.pkl` = training curve, `eval_data.pkl` = evaluation). Committed Lift models (1.7 cm base plate):

| Run (folder) | Cube | Evaluation success (100 episodes) |
|---|---|---|
| **`omp_lift_55mm_s2_best`** | 55 mm | **99 %** (reward 69.8), **use this one** |
| `omp_lift_55mm_s1` / `_s0` | 55 mm | 99 % / 50 % |
| `omp_lift_30mm_s0` / `_s1` / `_s2` | 30 mm | 100 % / 77 % / 85 % |

A model must be run with the cube size it was trained with: `Lift_omp.yaml` is set to 55 mm. For the 30 mm models, set
`cube_full_size: [0.03, 0.03, 0.03]`. Re-evaluate from `rlwp/` with
`python main.py test=True task=Lift_omp run_name=omp_lift_55mm_s2_best`.

Watch any saved model (waypoints chosen by the learned reward models, no exploration noise), one line per episode:
```bash
PYGLFW_LIBRARY_VARIANT=x11 python omp_tools/view_policy.py --task Lift_omp --run omp_lift_55mm_s2_best
```
`--force-open` overrides the last waypoint's gripper with "open" (where would a release land?).

More waypoints / warm start: a task config can set its own `num_wp` (e.g. `cfg/task/Stack3_omp.yaml`: grasp → carry →
lower and release). `train_omp.sh --start-wp K --init-from <run prefix>` trains only waypoints K.. and reuses each seed's
frozen waypoints 1..K-1 from `<run prefix>_s<seed>` (same task settings).

### 4.5 Running `main.py` directly

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
| `env.env_kwargs` | task YAML | extra `suite.make` arguments, e.g. an adapted task layout or object sizes (see below). Unset = the original call |

OM-P task configs: `cfg/task/{Lift,Stack,NutAssembly,PickPlace,Door}_omp.yaml`. They use the paper's schedules
(so the curves compare directly with Panda), the `default_open_manipulator_p` controller (coupled OSC), orientation
hold, position-only waypoints (`state_dim: 4`), and the workspace boxes from `check_omp_workspace.py`.

**The tasks are adapted to the OM-P on its 1.7 cm base plate.** With the gripper pointing down, the arm keeps the grip point
within ~0.83–0.905 m (3–10 cm above the table) and ~0.15–0.40 m from its base. rlwp moves in a straight line between
waypoints, and the OM-X fingers open sideways (along y). So objects are approached head-on (along x), and anything
carried sideways must start far enough from obstacles to rise over them. The original robosuite layouts don't fit
that. The `_omp` configs change them through `env.env_kwargs` (new environment arguments; their defaults reproduce the
original tasks, and Panda results are bit-identical):

| Task | What changed (`env_kwargs`) | Why |
|---|---|---|
| Lift_omp | `cube_full_size` 30 mm (stock 40 mm) | smallest cube with a safe grip margin: the gripper closes to ~22 mm. Set it to your real cube |
| Stack_omp | cubeA 40 mm onto cubeB 50 mm (the paper's sizes, set via `cubeA_full_size` / `cubeB_full_size`). cubeA spawns straight in front of the start pose (x −0.03…0.01, y −0.05…−0.01), cubeB 12–16 cm to the side (y 0.12…0.16). `relative_stack_check`: stacked = cubeA's bottom on cubeB's top, using the actual cube heights | sideways approach would push cubeA away with the fingers. cubeA must rise over cubeB before reaching it. The original check (cubeA centre > table + 5 cm) only fits the stock 40/50 mm cubes and misses real stacks on lower cubeBs |
| NutAssembly_omp | nut handle points at the robot (`nut_rotation` π), nut spawns at x 0.01…0.03, y ±0.03. Round peg at (0.02, −0.18), **3 cm tall** (was 13 cm). Square peg moved aside. Lift stage 5 cm | fingers would land on the ring before the handle. The nut can only be carried ~6 cm high |
| PickPlace_omp (can) | bins scaled 0.4× (0.16 × 0.20 m) with **1.5 cm lips** (were 8 cm), object bin straight ahead, target bin to the side. Robot on a stand at bin height. Lift stage 3 cm | original bins 0.6 m away with walls the can couldn't clear |
| Door_omp | door scaled 0.75×, handle stands off 1.5× further (bar ~10 cm off the panel), on a lower table (handle at 0.86 m) in line with the start pose. Robot on a stand. Position-only waypoints | original handle 27 cm above the base. The 13 cm-wide gripper housing must clear the panel |

Every adapted task is solvable with rlwp's 2-waypoint scheme inside its workspace box. `omp_tools/check_omp_tasks.py`
runs a hand-designed waypoint pair through rlwp's exact action rule: **20/20 successes for all five tasks**.

| Task | workspace low | workspace high |
|---|---|---|
| Lift_omp, Stack_omp, PickPlace_omp, Door_omp | [-0.20, -0.20, 0.825] | [0.05, 0.20, 0.905] |
| NutAssembly_omp (table at 0.82) | [-0.20, -0.25, 0.845] | [0.00, 0.25, 0.950] |

If you change the robot model, its base placement, the controller or a task layout, rerun
`omp_tools/check_omp_workspace.py` (regions per task in its `TASK_REGIONS`) and `omp_tools/check_omp_tasks.py`.

PickPlace needs rlwp's top-level `object`: `train_omp.sh` adds `object=can` for PickPlace tasks automatically. With
`main.py` directly, pass `object=can`.

## 6. Robot model summary

* **Arm**: `robosuite/robosuite/models/assets/robots/open_manipulator_p/robot.xml` (from ROBOTIS' URDF, real
  inertias, hardware joint ranges, Dynamixel-P torque limits). Class `OpenManipulatorP`.
* **Gripper**: `robosuite/robosuite/models/assets/grippers/omx_gripper.xml` (from ROBOTIS' OM-X MuJoCo model,
  20–80 mm opening, box fingerpads), class `OMXGripper`, on a 10 mm adapter plate.
* **Mount / base plate**: the robot stands on a **box-shaped base plate on the table top** (`OMPRiserMount`), by
  default the TurtleBot3 Waffle Pi plate: 0.14 × 0.14 m, **1.7 cm thick**. Base at x = −0.32 m, z = 0.817 m.
  **To change the box**, edit `BASE_PLATE_SIZE = (length x, width y, height z)` and, if needed,
  `BASE_PLATE_OFFSET = (x, y)` (plate centre relative to the robot base) at the top of
  `robosuite/robosuite/models/mounts/omp_riser_mount.py`. The XML file is only a template. With the gripper
  pointing down, OM-P's wrist (J5, ±90°) limits the grip point to ~0.10 m above the base, so the plate height shifts
  the whole vertical working range. After changing the height, rerun `check_omp_workspace.py` and
  `check_omp_tasks.py`, and update the workspace blocks (§5). Mirror the plate on the real robot.
* **Start pose**: gripper pointing down, 0.087 m above the table, 0.22 m in front of the base.
* **Controller**: `controllers/config/default_open_manipulator_p.json` = robosuite OSC_POSE with
  `uncouple_pos_ori: false`. The default uncoupled OSC moves this 6-DoF arm at half speed along x.

## 7. Troubleshooting

* `AttributeError: module 'robosuite' has no attribute 'make'`: you started Python *interactively or with `-c`* in
  `rl-waypoints/`, where the folder `robosuite/` shadows the package. Run from `rlwp/` or anywhere else (scripts run as
  files are fine).
* Viewer window problems on Wayland: prefix with `PYGLFW_LIBRARY_VARIANT=x11`.
* Loading the shipped Panda models on a CPU-only machine needs `map_location`, which is now built in.
* `run name 'test' is reserved`: choose another `-n`. The shipped Panda models are stored under `test`.

## 8. Object sizes (Lift / Stack)

Set the cubes to match your real objects in `rlwp/cfg/task/Lift_omp.yaml` (`cube_full_size`, now 30 mm) and
`Stack_omp.yaml` / `Stack3_omp.yaml` (`cubeA_full_size`, `cubeB_full_size`, now 40 / 50 mm): full x, y, z in metres (x = approach direction, y = across the fingers, z = height).
Cuboids are fine if they are always placed in the same orientation. From scripted tests (OM-P, 1.7 cm plate):
25–60 mm across the fingers (the gripper closes to ~22 mm and opens to ~80 mm), length along the approach up to ≥ 10 cm,
height ≥ 20 mm, Stack's bottom box ≤ ~55 mm tall. After changing sizes, run `omp_tools/check_omp_tasks.py`, and retrain.

For Stack, the size of the bottom cube matters most for *learning*: rlwp only learns to open the gripper over cubeB if
exploratory releases land on it often enough. With 30 mm cubes, training reached 35 % / 0 % / 48 % (2 waypoints) and
38 % / 0 % / 33 % (3 waypoints, which learned to hover over cubeB). A 50 mm cubeB gives ~2.8× the landing area.

