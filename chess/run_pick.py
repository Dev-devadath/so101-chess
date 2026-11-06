"""Run the learned PICK skill on the real arm, with IK around it.

For each FROM TO pair:  unpark -> IK hover above FROM -> gripper open ->
POLICY (descend, grasp, lift) until the done check -> IK carry to TO via the
raised travel pose -> lower until the piece's base (computed from the grasp
pose and the gripper reading) is PLACE_GAP_M above the square -> press it onto
the mat -> crack the jaws so it settles upright -> rise -> open -> hover ->
next pair / park.

Done check: gripper closed on something (GRIPPER_EMPTY < g < GRIPPER_CLOSED) for
DONE_HOLD_S and the tip risen DONE_LIFT_M above the grasp height. A gripper that
closes on nothing is a miss: reopen, back to the hover, run the policy again
(--retries). Safety stops: --max-s per attempt, tip below the mat, or a command
more than MAX_JUMP_DEG from the current pose.

Usage:  .venv/bin/python chess/run_pick.py d2 d4 [c5 c4 ...] [--policy policies/act_pick_v1] [--max-s 15]
        .venv/bin/python chess/run_pick.py d2 d4 --policy outputs/act_<dataset> --ckpt 080000   a training output instead
        .venv/bin/python chess/run_pick.py d2 d4 --no-place    stop after the policy, hold the piece, park
--jaw-roll 0 for policies trained on so101_pick / so101_pick_trim (recorded with sideways jaws).
Calibrate the board first (calibrate_board.py): the top camera is warped exactly as in record_pick.py.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.common.control_utils import predict_action
from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame

from arm import JOINTS, OFFSET, Kinematics, load_arm_calib, move_to, park, read_joints, send_joints, unpark
from hover_squares import hover_plan, square_target
from reach_map import JAW_ROLL_DEG, calib_roll, held_piece_tool, held_tool, solve_point
from record_pick import GRIPPER_OPEN, TASK, connect_with_cameras, dataset_features
from teach_squares import POINTS_PATH

ROOT = Path(__file__).parent.parent
POLICY = "policies/act_pick_v1"  # pretrained_model folder (relative to the repo root)
DEVICE = "mps"
FPS = 30
GRIPPER_CLOSED = 30.0  # with the pads, king/queen held at the widest part read up to 24 (open is 40)
GRIPPER_EMPTY = 7.0  # jaws met, nothing in them: 5.0 with the TPU pads (2 bare); the thinnest held piece reads 8.9
DONE_HOLD_S = 0.5
DONE_LIFT_M = 0.015  # tip risen this far above the grasp height counts as lifted (demos end 15-40 mm up)
MAX_JUMP_DEG = 30.0  # a single command further than this from the current joint reading aborts
PLACE_GAP_M = 0.002  # piece base this far above the mat at the end of the descent
# Guided release: a piece held higher or more tilted than modelled still has to land on its base.
PLACE_PRESS_M = 0.003  # then press the modelled base this far INTO the mat (open-loop ramp, servos just stall a little)
RELEASE_CRACK = 4.0  # open the jaws only this much first: the piece settles upright between the grooves
PLACE_RISE_M = 0.012  # rise this far with the jaws cracked before opening wide, so the pads cannot drag it


LOG_DIR = Path(__file__).parent / "run_logs"


def load_policy(path=POLICY, ckpt="last", ensemble=False):
    path = ROOT / path
    if (path / "checkpoints").is_dir():  # a lerobot training output: pick a checkpoint
        path = path / "checkpoints" / ckpt / "pretrained_model"
    policy = ACTPolicy.from_pretrained(str(path)).to(DEVICE).eval()
    if ensemble:
        # ACT temporal ensembling: predict a chunk every frame, average the overlapping predictions
        policy.config.temporal_ensemble_coeff = 0.01
        policy.config.n_action_steps = 1
        policy.temporal_ensembler = ACTTemporalEnsembler(0.01, policy.config.chunk_size)
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=str(path), preprocessor_overrides={"device_processor": {"device": DEVICE}}
    )
    return policy, pre, post


def infer(policy, pre, post, feats, obs):
    """Robot observation dict (joints + 'wrist' + 'top' images) -> action dict in lerobot degrees."""
    frame = build_dataset_frame(feats, obs, OBS_STR)
    a = predict_action(frame, policy, torch.device(DEVICE), pre, post, use_amp=False, task=TASK, robot_type="so101_follower")
    return make_robot_action(a, feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("squares", nargs="+", help="FROM TO [FROM TO ...]")
    ap.add_argument("--policy", default=POLICY, help="pretrained_model folder, or a training output folder under outputs/")
    ap.add_argument("--ckpt", default="last", help="checkpoint when --policy is a training output")
    ap.add_argument("--jaw-roll", type=float, default=JAW_ROLL_DEG, help="wrist roll offset at the hover (0 = sideways jaws, as in so101_pick)")
    ap.add_argument("--max-s", type=float, default=15.0)
    ap.add_argument("--retries", type=int, default=2, help="extra policy attempts after the gripper closes on nothing")
    ap.add_argument("--no-place", action="store_true")
    ap.add_argument("--ensemble", action="store_true", help="temporal ensembling (re-predict every frame) instead of open-loop chunks")
    ap.add_argument("--release", type=float, default=15.0, help="open this much beyond the held reading when releasing (keeps the jaws off the neighbours)")
    a = ap.parse_args()
    if len(a.squares) % 2:
        ap.error("need FROM TO pairs")
    pairs = list(zip(a.squares[::2], a.squares[1::2]))

    kin = Kinematics()
    calib = load_arm_calib()
    tool = calib.get("tool_offset", np.zeros(3))
    roll = calib_roll(calib)
    kin.set_roll(roll)
    taught = json.load(open(POINTS_PATH))
    n = calib["normal"]
    # hover with the centre of the OPEN jaw gap over the square (the fixed jaw is at the tool point)
    plans = {sq: hover_plan(kin, calib, taught, held_tool(tool, GRIPPER_OPEN), roll, sq, a.jaw_roll) for p in pairs for sq in p}

    policy, pre, post = load_policy(a.policy, a.ckpt, a.ensemble)
    print(f"policy {a.policy} loaded on {DEVICE}; chunk {policy.config.chunk_size}, n_action_steps {policy.config.n_action_steps}, ensemble {a.ensemble}")
    LOG_DIR.mkdir(exist_ok=True)
    run_id = time.strftime("%m%d_%H%M")
    robot, top = connect_with_cameras()
    feats = dataset_features(robot)
    try:
        input(f"\n{len(pairs)} move(s): {' '.join(f'{s}->{d}' for s, d in pairs)}. Pieces in place? Hand on the switch? Enter to go... ")
        first = True
        for src, dst in pairs:
            q_h, h_hover, _, _ = plans[src]
            base = square_target(taught, calib, src)
            height = lambda q_raw: float((kin.tool_pos(np.asarray(q_raw) - OFFSET, tool) - base) @ n)
            if first:
                unpark(robot, q_h)
                first = False
            else:
                move_to(robot, q_h, duration_s=2.0)
            for attempt in range(1 + a.retries):
                send_joints(robot, read_joints(robot), gripper=GRIPPER_OPEN)
                time.sleep(0.7)
                policy.reset()
                print(f">> policy on {src}  (hover {h_hover * 100:.0f} cm, attempt {attempt + 1})")
                t0, closed_since, lowest, reason, saved, q_close = time.perf_counter(), None, np.inf, "timeout", False, None
                trace = []  # (t, 6 joint readings, 6 commanded) per frame, saved for offline comparison with the demos
                while time.perf_counter() - t0 < a.max_s:
                    tl = time.perf_counter()
                    obs = robot.get_observation()
                    obs["top"] = top.read()
                    if not saved:  # what the policy saw first, for comparison with the training frames
                        cv2.imwrite(str(LOG_DIR / f"{run_id}_{src}_top.jpg"), cv2.cvtColor(obs["top"], cv2.COLOR_RGB2BGR))
                        cv2.imwrite(str(LOG_DIR / f"{run_id}_{src}_wrist.jpg"), cv2.cvtColor(obs["wrist"], cv2.COLOR_RGB2BGR))
                        saved = True
                    q_raw = np.array([obs[f"{j}.pos"] for j in JOINTS])
                    tip_h = height(q_raw)
                    lowest = min(lowest, tip_h)
                    g = obs["gripper.pos"]
                    if g < GRIPPER_CLOSED:
                        if closed_since is None:
                            closed_since, q_close = time.perf_counter(), q_raw - OFFSET  # pose at the grasp
                    else:
                        closed_since = None
                    held = closed_since and time.perf_counter() - closed_since > DONE_HOLD_S
                    if held and g < GRIPPER_EMPTY:
                        reason = "missed: gripper closed on nothing"
                        break
                    if held and tip_h > lowest + DONE_LIFT_M:
                        reason = "done"
                        break
                    if tip_h < -0.01:
                        reason = "tip below the mat"
                        break
                    act = infer(policy, pre, post, feats, obs)
                    # the servos clamp to their calibrated range themselves; guard only against a wild output
                    jump = {j: act[f"{j}.pos"] - obs[f"{j}.pos"] for j in JOINTS if abs(act[f"{j}.pos"] - obs[f"{j}.pos"]) > MAX_JUMP_DEG}
                    if jump:
                        reason = f"command jumps too far from the current pose: { {j: round(v) for j, v in jump.items()} }"
                        break
                    robot.send_action(act)
                    trace.append([time.perf_counter() - t0, *q_raw, g, *(act[f"{j}.pos"] for j in JOINTS), act["gripper.pos"]])
                    print(f"\r   t {time.perf_counter() - t0:4.1f}s  tip {tip_h * 1000:4.0f} mm  gripper {g:4.0f} -> {act['gripper.pos']:4.0f}   ", end="", flush=True)
                    dt = time.perf_counter() - tl
                    if dt < 1 / FPS:
                        time.sleep(1 / FPS - dt)
                g_end = robot.get_observation()["gripper.pos"]
                np.save(LOG_DIR / f"{run_id}_{src}_{attempt + 1}_trace.npy", np.array(trace))
                print(f"\n   stopped: {reason} after {time.perf_counter() - t0:.1f} s; lowest tip {lowest * 1000:.0f} mm, gripper {g_end:.0f}, top-cam stalls {top.stalls}")
                if reason == "done" or not reason.startswith("missed"):
                    break
                # missed: open where we are (don't drag a nudged piece), back to the hover, try again
                send_joints(robot, read_joints(robot), gripper=GRIPPER_OPEN)
                time.sleep(0.5)
                move_to(robot, q_h, duration_s=1.5)
            move_to(robot, q_h, duration_s=1.0)  # back to the exact hover
            if a.no_place or reason != "done":
                print("   holding here; parking")
                break
            q_h2, _, _, r2 = plans[dst]
            base2 = square_target(taught, calib, dst)
            # place the piece's BASE (half a jaw gap from the touch point, hanging below the
            # fingers by however high it was grasped) PLACE_GAP_M above the square centre
            piece = held_piece_tool(kin, q_close, g_end, tool, base, n)
            kin.set_roll(r2)
            q_p, err, _, badp = solve_point(kin, base2 + PLACE_GAP_M * n, -n, piece, r2, q_warm=q_h2)
            if badp or err > 0.005:
                print(f"   {dst}: place IK failed; parking with the piece")
                break
            unpark(robot, q_h2)  # carry via the raised travel pose, clear of the other pieces
            move_to(robot, q_p, duration_s=2.0)
            # guided release: press onto the mat (no error trimming, so a stalled servo is not pushed
            # harder), crack the jaws so the piece settles on its base, rise, then open wide
            q_t, err_t, _, bad_t = solve_point(kin, base2 - PLACE_PRESS_M * n, -n, piece, r2, q_warm=q_p)
            if not bad_t and err_t <= 0.005:
                q_now = read_joints(robot)
                for k in range(1, 16):
                    send_joints(robot, q_now + (q_t - q_now) * k / 15)
                    time.sleep(1 / FPS)
                time.sleep(0.3)
            send_joints(robot, read_joints(robot), gripper=g_end + RELEASE_CRACK)
            time.sleep(0.5)
            q_u, err_u, _, bad_u = solve_point(kin, base2 + PLACE_RISE_M * n, -n, piece, r2, q_warm=q_p)
            move_to(robot, q_u if not bad_u and err_u <= 0.005 else q_p, duration_s=0.8)
            send_joints(robot, read_joints(robot), gripper=g_end + a.release)
            time.sleep(0.5)
            move_to(robot, q_h2, duration_s=1.2)
            print(f"   placed on {dst} (held at {g_end:.0f}, piece base {np.linalg.norm(piece - tool) * 1000:.0f} mm from the fingertips)")
    finally:
        try:
            park(robot)
        finally:
            top.disconnect()
            robot.disconnect()


if __name__ == "__main__":
    main()
