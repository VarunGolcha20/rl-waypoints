import os
import random

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import robosuite
from robosuite.controllers import load_controller_config


def set_seed(seed):
    """Seeds python, numpy (also used by robosuite's object placement) and torch. None keeps runs unseeded."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_save_name(env, robot, run_name, object=None, use_latch=False):
    """Model directory under models/. Panda keeps the original layout; other robots get a robot prefix."""
    if object is None:
        save_name = env + '/' + run_name
        if env == 'Door':
            save_name = env + ('/with_latch/' if use_latch else '/without_latch/') + run_name
    else:
        save_name = env + '/' + object + '/' + run_name
    if robot != 'Panda':
        save_name = robot + '/' + save_name
    return save_name


def get_controller_config(env_cfg):
    """robosuite controller config: the generic OSC_POSE (original behaviour), or a named json from
    robosuite/controllers/config/ when the task YAML sets env.controller."""
    name = env_cfg.get('controller')
    if name is None:
        return load_controller_config(default_controller="OSC_POSE")
    path = os.path.join(os.path.dirname(robosuite.__file__), 'controllers', 'config', name + '.json')
    return load_controller_config(custom_fpath=path)


def get_env_kwargs(env_cfg, object, use_latch):
    """Extra suite.make arguments: rlwp's original ones, overridden / extended by the task YAML's env.env_kwargs
    (e.g. an adapted task layout for a smaller robot). Without env_kwargs this is exactly the original call."""
    def plain(v):  # hydra ListConfig / DictConfig -> list / dict, recursively
        if isinstance(v, (list, tuple)) or type(v).__name__ == 'ListConfig':
            return [plain(x) for x in v]
        if isinstance(v, dict) or type(v).__name__ == 'DictConfig':
            return {k: plain(x) for k, x in v.items()}
        return v

    kwargs = dict(single_object_mode=2, object_type=object, use_latch=use_latch)
    kwargs.update({k: plain(v) for k, v in (env_cfg.get('env_kwargs') or {}).items()})
    return kwargs


def get_workspace(env_cfg, state):
    """Waypoint position bounds relative to the start state, from the task YAML's absolute env.workspace
    (low / high, world frame). None when no workspace is configured (original behaviour).
    With env.workspace.radial (center [x, y] world, r_min, r_max) the result gets a third element, the reach
    constraint relative to the start state: (center_xy, r_min, r_max). Waypoints must then also lie between r_min and
    r_max (horizontally) from the center, e.g. an arm's base axis: a box alone can't describe a ring-shaped reach."""
    ws = env_cfg.get('workspace')
    if ws is None:
        return None
    lo, hi = np.array(ws['low']) - state[:3], np.array(ws['high']) - state[:3]
    rad = ws.get('radial')
    if rad is None:
        return lo, hi
    return lo, hi, (np.array(rad['center'], dtype=float) - state[:2], float(rad['r_min']), float(rad['r_max']))


def radial_violation(wp, radial):
    """how far (m) a waypoint's xy lies outside the reach ring (0 inside)"""
    if radial is None:
        return 0.0
    c, r_min, r_max = radial
    r = np.linalg.norm(np.asarray(wp[:2]) - c)
    return max(0.0, r_min - r, r - r_max)


def radial_project(wp, radial):
    """move a waypoint's xy radially onto the nearest point of the reach ring (other dims unchanged)"""
    wp = np.array(wp, dtype=float)
    if radial is None:
        return wp
    c, r_min, r_max = radial
    d = wp[:2] - c
    r = np.linalg.norm(d)
    if r < 1e-9:
        d, r = np.array([1.0, 0.0]), 1.0
    wp[:2] = c + d / r * np.clip(r, r_min, r_max)
    return wp


def clip_waypoint(wp, lb, ub, radial=None):
    """box clip, then (if a reach ring is set) radial projection and box clip again"""
    wp = np.clip(wp, lb, ub)
    if radial is not None:
        for _ in range(3):
            wp = np.clip(radial_project(wp, radial), lb, ub)
    return wp


def orientation_hold_action(obs, ori_goal):
    """OSC delta-rotation action (in [-1, 1], 0.5 rad per unit) that turns the gripper back to ori_goal."""
    rotvec = (ori_goal * R.from_quat(obs['robot0_eef_quat']).inv()).as_rotvec()
    return np.clip(rotvec / 0.5, -1., 1.)


def bounded_solution(res, xi0, lb, ub, cost_fn, tol=1e-6, radial=None):
    """Guard an SLSQP waypoint solution. SLSQP enforces the box only as a linear constraint and can run away (seen:
    waypoints 1e7 m out of bounds when the learned reward keeps rising outside the box). A non-finite solution is
    replaced by its (clipped) start point, an out-of-bounds one is clipped; both get their cost re-evaluated. A valid
    solution (the normal case) is returned unchanged, so runs that never hit this behave exactly as before."""
    x = res.x
    if not np.all(np.isfinite(x)):
        x = clip_waypoint(xi0, lb, ub, radial)
        return x, cost_fn(x)
    if np.any(x < lb - tol) or np.any(x > ub + tol) or radial_violation(x, radial) > 1e-4:
        x = clip_waypoint(x, lb, ub, radial)
        return x, cost_fn(x)
    return x, res.fun


def lift_first_error(error, wp_idx, enabled, tol=0.01):
    """env.lift_first: in every segment after the first, while the waypoint is more than `tol` above the gripper, track
    only its height (zero the horizontal error), then move horizontally. rlwp's rule 10 * error saturates per axis, so
    a long sideways move otherwise outruns the rise and a carried object skims what lies between (Stack: cubeA clipped
    cubeB's top edge). Off (default) = the original straight-line rule."""
    if enabled and wp_idx >= 1 and error[2] > tol:
        error = np.array(error, dtype=float)
        error[:2] = 0.0
    return error
