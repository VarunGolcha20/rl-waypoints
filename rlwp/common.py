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


def get_workspace(env_cfg, state):
    """Waypoint position bounds relative to the start state, from the task YAML's absolute env.workspace
    (low / high, world frame). None when no workspace is configured (original behaviour)."""
    ws = env_cfg.get('workspace')
    if ws is None:
        return None
    return np.array(ws['low']) - state[:3], np.array(ws['high']) - state[:3]


def orientation_hold_action(obs, ori_goal):
    """OSC delta-rotation action (in [-1, 1], 0.5 rad per unit) that turns the gripper back to ori_goal."""
    rotvec = (ori_goal * R.from_quat(obs['robot0_eef_quat']).inv()).as_rotvec()
    return np.clip(rotvec / 0.5, -1., 1.)
