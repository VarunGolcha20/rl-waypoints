"""
Watch a trained (or still training) rlwp model in MuJoCo's viewer: each episode picks the waypoints with the saved
reward models (as rlwp's evaluation does, no exploration noise) and executes them with rlwp's action rule.
Read-only: loads rlwp/models/..., never writes.

    PYGLFW_LIBRARY_VARIANT=x11 python omp_tools/view_policy.py --task Stack_omp --run omp_stack_30mm_s2
    PYGLFW_LIBRARY_VARIANT=x11 python omp_tools/view_policy.py --task Lift_omp --run omp_lift_30mm_s0
    PYGLFW_LIBRARY_VARIANT=x11 python omp_tools/view_policy.py --task Lift_omx --run omx_lift_s2 --seed 1000

Prints one line per episode (grasp / lift / success). Close the window to quit.

Episode controls (in the viewer window): Left arrow = previous episode, Right arrow = next episode, Up arrow = replay the
current episode. --episode N starts at episode N. Replays are exact: each episode's object placement and waypoints are
regenerated from the random-number state recorded at its start, so episode N is always the same for a given --seed.
--force-open replaces the last waypoint's gripper command with "open", to see where a release would land.

Real vs false success (Lift / Stack): robosuite's success flag is compared with what physically happened, and a marker
with a label floats above the scene: grey = running, yellow = robosuite reports success, then at the end of the episode
green = REAL SUCCESS, red = FALSE SUCCESS (robosuite says success, but the cube never rose 4 cm above its resting height
(Lift) / cubeA is not resting on cubeB at the end (Stack)), grey = FAIL. Running totals in the terminal.
"""
import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
RLWP = os.path.join(HERE, "..", "rlwp")
sys.path.insert(0, RLWP)
os.chdir(RLWP)  # rlwp loads models from ./models
from common import get_controller_config, get_env_kwargs, get_save_name, get_workspace, lift_first_error, orientation_hold_action  # noqa: E402
from eval import Method  # noqa: E402

import robosuite as suite  # noqa: E402

OBJECTS = {"Lift": ["cube_pos"], "Stack": ["cubeA_pos", "cubeB_pos"], "NutAssembly": ["RoundNut_pos"],
           "PickPlace": ["can_pos"], "Door": ["door_pos", "handle_pos"]}
GRASP_TARGET = {"Lift": "cube", "Stack": "cubeA"}
GREY, YELLOW, GREEN, RED = [0.6, 0.6, 0.6, 0.9], [1.0, 0.85, 0.1, 1.0], [0.1, 0.85, 0.2, 1.0], [0.9, 0.1, 0.1, 1.0]


def set_marker(viewer, pos, rgba, text):
    """coloured sphere + text label drawn into the viewer's scene (user_scn); MuJoCo labels hold at most 99 characters"""
    text = text[:99]
    with viewer.lock():
        g = viewer.user_scn.geoms[0]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.012, 0, 0]), np.asarray(pos, dtype=float),
                            np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
        g.label = text
        viewer.user_scn.ngeom = 1


def lifted_for_real(env, obs):
    """Lift: cube centre >= 4 cm above its resting height (robosuite only asks centre > table + 4 cm)"""
    return obs["cube_pos"][2] >= env.table_offset[2] + env.cube.size[2] + 0.04


def stacked_for_real(env, obs):
    """Stack: cubeA rests on cubeB (bottom within 6 mm of cubeB's top, centre over cubeB) and is not held"""
    a, b = obs["cubeA_pos"], obs["cubeB_pos"]
    on = abs((a[2] - env.cubeA.size[2]) - (b[2] + env.cubeB.size[2])) < 0.006 and np.all(np.abs(a[:2] - b[:2]) < env.cubeB.size[:2])
    return on and not env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.cubeA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Stack_omp", help="rlwp task config (cfg/task/<task>.yaml)")
    ap.add_argument("--run", required=True, help="run name, e.g. omp_stack_30mm_s2")
    ap.add_argument("--object", default="can", help="PickPlace object")
    ap.add_argument("--force-open", action="store_true",
                    help="override the last waypoint's gripper command with 'open' (what if it released there?)")
    ap.add_argument("--seed", type=int, default=None, help="seed numpy (object placements), e.g. 1000 = rlwp eval seed 0")
    ap.add_argument("--pause", type=float, default=2.0, help="seconds to hold the final state / verdict after each episode")
    ap.add_argument("--episode", type=int, default=1, help="start at this episode (earlier ones are generated, not shown)")
    ap.add_argument("--n-inits", type=int, default=20, help="random restarts of the waypoint optimiser (rlwp evaluation "
                                                             "uses 5; 20 gives stable answers for Stack's large landing area)")
    args = ap.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    task = yaml.safe_load(open(os.path.join("cfg", "task", args.task + ".yaml")))
    name, ec = task["name"], task["env"]
    obj = args.object if name == "PickPlace" else None
    env = suite.make(name, robots=ec["robot"], controller_configs=get_controller_config(ec), has_renderer=False,
                     has_offscreen_renderer=False, use_camera_obs=False, reward_shaping=True, control_freq=10,
                     initialization_noise=None, hard_reset=False, **get_env_kwargs(ec, obj, ec["use_latch"]))
    rng_reset0 = np.random.get_state()
    obs = env.reset()
    get_objs = lambda o: np.concatenate([o[k] for k in OBJECTS[name]])
    state = lambda o: o["robot0_eef_pos"]
    save_name = get_save_name(name, ec["robot"], args.run, obj, ec["use_latch"])
    agent = Method(state_dim=ec["state_dim"], objs=get_objs(obs), wp_id=2, save_name=save_name,
                   config={"task": task, "n_inits": args.n_inits}, workspace=get_workspace(ec, state(obs)))
    wp_steps, gripper_steps = ec["wp_steps"], ec["gripper_steps"]
    print("loaded models/%s" % save_name, flush=True)
    print("controls: Left = previous episode, Right = next episode, Up = replay current episode", flush=True)

    # Exact replay: an episode is fully determined by numpy's random state before its reset (object placement) and
    # before its waypoint optimisation (the frozen reward models are fixed), so both are recorded per episode.
    history = [{"rng_reset": rng_reset0, "rng_traj": np.random.get_state()}]

    def start_of(k):
        """reset the env and the RNG to the start of episode k (1-based, already in history); returns its observation"""
        h = history[k - 1]
        np.random.set_state(h["rng_reset"])
        o = env.reset()
        np.random.set_state(h["rng_traj"])
        return o

    def goto(k):
        """start of episode k; unseen episodes are generated on the way, in the original order (reset -> waypoint
        optimisation -> next reset), without simulating them: the physics draws no random numbers"""
        while len(history) < k:
            n = len(history)
            agent.traj_opt(n, get_objs(start_of(n)))
            rng_reset = np.random.get_state()
            env.reset()
            history.append({"rng_reset": rng_reset, "rng_traj": np.random.get_state()})
        return start_of(k)

    command = {"key": None}

    def on_key(keycode):
        command["key"] = {263: "prev", 262: "next", 265: "replay"}.get(keycode, command["key"])

    with mujoco.viewer.launch_passive(env.sim.model._model, env.sim.data._data, key_callback=on_key) as viewer:
        if ec["robot"] == "OpenManipulatorX":
            viewer.cam.lookat[:], viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = (-0.06, 0.04, 0.86), 0.65, 145, -25
        else:
            viewer.cam.lookat[:], viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = (-0.12, 0.05, 0.85), 1.0, 145, -25
        with viewer.lock():
            viewer.opt.geomgroup[0] = 0
        marker = state(obs) + [0.0, 0.06, 0.09]  # above the work area, beside the start pose
        verdicts = {}  # episode -> verdict (replays don't double-count)
        judged = name in ("Lift", "Stack")
        episode = max(1, args.episode)
        obs = goto(episode)
        while viewer.is_running():
            command["key"] = None
            traj = agent.traj_opt(episode, get_objs(obs)).reshape(2, ec["state_dim"])
            if args.force_open:
                traj[-1, 3] = -1.0
            targets = traj[:, :3] + state(obs)
            goal = R.from_quat(obs["robot0_eef_quat"])
            time_s, grasped, success, reward, first, real = 0, False, False, 0.0, None, False
            if judged:
                set_marker(viewer, marker, GREY, "episode %d" % episode)
            for t in range(2 * wp_steps):
                if not viewer.is_running():
                    return
                if command["key"] is not None:  # jump requested mid-episode
                    break
                t0 = time.time()
                i = t // wp_steps
                err = lift_first_error(targets[i] - state(obs), i, ec.get("lift_first", False))
                if t < 10:
                    a = np.r_[10 * err, 0, 0, 0, -1.0]
                elif time_s >= wp_steps - gripper_steps:
                    a = np.r_[np.zeros(6), traj[i, 3:]]
                else:
                    a = np.r_[10 * err, 0, 0, 0, 0.0]
                if ec.get("orientation_hold"):
                    a[3:6] = orientation_hold_action(obs, goal)
                time_s = time_s + 1 if time_s + 1 < wp_steps else 1
                obs, r, _, _ = env.step(a)
                reward += r
                if env._check_success() and not success and judged:
                    first = t
                    set_marker(viewer, marker, YELLOW, "robosuite: success (step %d)" % t)
                success |= env._check_success()
                if name == "Lift" and lifted_for_real(env, obs):
                    real = True
                if t == wp_steps - 1 and name in GRASP_TARGET:
                    grasped = env._check_grasp(gripper=env.robots[0].gripper, object_geoms=getattr(env, GRASP_TARGET[name]))
                viewer.sync()
                time.sleep(max(0.0, env.control_timestep - (time.time() - t0)))
            interrupted = command["key"] is not None
            print("episode %d%s: waypoints %s, gripper %s | grasped after wp1: %s | success: %s | reward %.1f" % (
                episode, " (interrupted)" if interrupted else "", np.round(targets, 3).tolist(), np.round(traj[:, 3], 2).tolist(),
                grasped, success, reward), flush=True)
            if judged and not interrupted:
                if name == "Stack":
                    real = stacked_for_real(env, obs)
                why = {"Lift": "cube never rose 4 cm", "Stack": "cubeA not on cubeB at the end"}[name]
                if success and real:
                    verdict, col, label = "REAL SUCCESS", GREEN, "REAL SUCCESS"
                elif success:
                    verdict, col, label = "FALSE SUCCESS", RED, "FALSE SUCCESS: " + why
                else:
                    verdict, col, label = "FAIL", GREY, "FAIL"
                if name == "Stack":  # why the end state passes / fails, in terms of the strict check's quantities
                    a, b = obs["cubeA_pos"], obs["cubeB_pos"]
                    detail = "centre offset %.0f mm, cubeA tilt %.0f deg, on cubeB %s" % (
                        1000 * np.linalg.norm(a[:2] - b[:2]), env._flat_tilt(env.sim.data.body_xmat[env.cubeA_body_id].reshape(3, 3)),
                        "yes" if stacked_for_real(env, obs) else "no")
                    if getattr(env, "stack_check", "original") == "strict":
                        detail += " (strict: centre <= %.0f mm, tilt <= %.0f deg)" % (1000 * env.stack_center_tol, env.stack_max_tilt)
                    label += " | off-centre %.0f mm, tilt %.0f deg" % (
                        1000 * np.linalg.norm(a[:2] - b[:2]), env._flat_tilt(env.sim.data.body_xmat[env.cubeA_body_id].reshape(3, 3)))
                    print("   end state: " + detail, flush=True)
                verdicts[episode] = verdict
                set_marker(viewer, marker, col, "ep %d: %s" % (episode, label))
                tally = {v: sum(x == v for x in verdicts.values()) for v in ("REAL SUCCESS", "FALSE SUCCESS", "FAIL")}
                print("   -> %s (robosuite success %s%s; real %s)   totals over %d distinct episodes: %s" % (
                    verdict, success, " at step %d" % first if first is not None else "", real, len(verdicts),
                    ", ".join("%s %d" % kv for kv in tally.items())), flush=True)
            t_end = time.time()
            while viewer.is_running() and time.time() - t_end < args.pause and command["key"] is None:  # hold the final state
                viewer.sync()
                time.sleep(0.05)
            key = command["key"] or "next"
            episode = max(1, episode - 1) if key == "prev" else episode if key == "replay" else episode + 1
            obs = goto(episode)
            viewer.sync()


if __name__ == "__main__":
    main()
