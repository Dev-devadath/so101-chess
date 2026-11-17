"""Run the learned PICK skill on the real arm, with IK around it.

For each FROM TO pair:  unpark -> IK hover above FROM -> gripper open ->
POLICY (descend, grasp, lift) until the done check -> glide to TO (up to the
clearance height, straight across, down to the hover) -> lower until the piece's base (computed from the grasp
pose and the gripper reading) is PLACE_GAP_M above the square -> press it onto
the mat -> crack the jaws so it settles upright -> open -> rise -> hover ->
next pair / park.

Done check: gripper closed on something (GRIPPER_EMPTY < g < GRIPPER_CLOSED) for
DONE_HOLD_S and the tip risen DONE_LIFT_M above the grasp height. A gripper that
closes on nothing is a miss: reopen, back to the hover, run the policy again
(--retries). Safety stops: --max-s per attempt, tip below the mat, or a command
more than MAX_JUMP_DEG from the current pose.

The whole sequence is the Mover class, which play.py drives for a game.

Usage:  .venv/bin/python chess/run_pick.py d2 d4 [c5 c4 ...] [--policy policies/act_pick_v1] [--max-s 15]
        .venv/bin/python chess/run_pick.py d2 d4 --policy outputs/act_<dataset> --ckpt 080000   a training output instead
        .venv/bin/python chess/run_pick.py d2 d4 --no-place    stop after the policy, hold the piece, park
        .venv/bin/python chess/run_pick.py e5 drop             carry a piece to the off-board drop zone
--jaw-roll 0 for policies trained on so101_pick / so101_pick_trim (recorded with sideways jaws).
Calibrate the board first (calibrate_board.py): the top camera is warped exactly as in record_pick.py.
"""

import argparse
import json
import sys
import threading
import time
from contextlib import contextmanager
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
from hover_squares import DROP, add_drop_zone, hover_plan, square_target
from reach_map import JAW_ROLL_DEG, calib_roll, held_piece_tool, held_tool, solve_point
from record_pick import GRIPPER_OPEN, TASK, connect_with_cameras, dataset_features, jittered_hover
from teach_squares import POINTS_PATH

ROOT = Path(__file__).parent.parent
POLICY = "policies/act_pick_v1"  # pretrained_model folder (relative to the repo root)
DEVICE = "mps"
FPS = 30
GRIPPER_CLOSED = 30.0  # with the pads, king/queen held at the widest part read up to 24 (open is 40)
GRIPPER_EMPTY = 7.0  # jaws met, nothing in them: 5.0 with the TPU pads (2 bare); the thinnest held piece reads 8.9
DONE_HOLD_S = 0.5
SQUEEZE_LIFT_M = 0.005  # jaws closed and the tip risen this far: the grasp is decided, so the gripper is commanded fully shut
DONE_LIFT_M = 0.015  # tip risen this far above the grasp height counts as lifted (demos end 15-40 mm up)
MAX_JUMP_DEG = 30.0  # a single command further than this from the current joint reading aborts
PLACE_GAP_M = 0.002  # piece base this far above the mat at the end of the descent
# Guided release: a piece held higher or more tilted than modelled still has to land on its base.
PLACE_PRESS_M = 0.003  # then press the modelled base this far INTO the mat (open-loop ramp, servos just stall a little)
RELEASE_CRACK = 4.0  # open the jaws only this much first: the piece settles upright between the grooves
DROP_GAP_M = 0.03  # captured pieces are simply let go this far above the drop zone
# Moves between squares glide: rise to a clearance height, straight line across the board,
# descend onto the hover. Whatever hangs below the fingertips (open jaw tips, or the base of
# a carried piece) stays CLEAR_M above the mat; near the base, where that does not solve,
# the path steps down early. Falls back to the raised travel pose when a waypoint fails.
CLEAR_M = 0.10
GLIDE_STEP_M = 0.025
GLIDE_SPEED = 0.12  # m/s along the path
# The policy may steer sideways only this far: the commanded pan keeps the centre of the open jaw gap within
# this distance of the square centre, measured along the pan swing. The demos grasp within 17 mm (1st-99th
# percentile -14..+17, median 0); in play the picks that held were within 18 and the sideways misses at 19-58.
SIDE_LIM_M = 0.015
RETRY_SHIFT_MM = 6.0  # a retry after a miss starts from the hover shifted by up to this much (the demos start within 6 mm)
CARRY_FPS = 15  # the carry-and-place film (cameras only, half size)
GLIDE_MAX_JUMP_DEG = 30.0  # a waypoint further than this (any joint but roll) from the previous is an IK branch jump


LOG_DIR = Path(__file__).parent / "run_logs"


def load_policy(path=POLICY, ckpt="last", ensemble=False):
    """(policy, preprocessor, postprocessor, relative). relative: trained on a scripts/make_relative.py
    dataset (name ending in _rel), so its arm joints are measured from the pose the policy starts at."""
    path = ROOT / path
    if (path / "checkpoints").is_dir():  # a lerobot training output: pick a checkpoint
        path = path / "checkpoints" / ckpt / "pretrained_model"
    cfg = path / "train_config.json"
    relative = cfg.exists() and json.load(open(cfg))["dataset"]["repo_id"].endswith("_rel")
    policy = ACTPolicy.from_pretrained(str(path)).to(DEVICE).eval()
    if ensemble:
        # ACT temporal ensembling: predict a chunk every frame, average the overlapping predictions
        policy.config.temporal_ensemble_coeff = 0.01
        policy.config.n_action_steps = 1
        policy.temporal_ensembler = ACTTemporalEnsembler(0.01, policy.config.chunk_size)
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=str(path), preprocessor_overrides={"device_processor": {"device": DEVICE}}
    )
    return policy, pre, post, relative


def infer(policy, pre, post, feats, obs):
    """Robot observation dict (joints + 'wrist' + 'top' images) -> action dict in lerobot degrees."""
    frame = build_dataset_frame(feats, obs, OBS_STR)
    a = predict_action(frame, policy, torch.device(DEVICE), pre, post, use_amp=False, task=TASK, robot_type="so101_follower")
    return make_robot_action(a, feats)


def debug_frame(obs, t, tip_h, cmd_h, g, g_cmd, label=None):
    """Wrist and top camera side by side (BGR) with time, tip height (actual / commanded) and gripper burnt in.

    With a label (the place snapshots) the text is just the label and the gripper reading (g None: the label alone)."""
    img = np.hstack([obs["wrist"], cv2.resize(obs["top"], (obs["wrist"].shape[0],) * 2)])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    txt = label if g is None else f"{label}  gripper {g:4.1f}" if label else f"t {t:4.1f}s  tip {tip_h * 1000:4.0f} mm (cmd {cmd_h * 1000:4.0f})  gripper {g:4.1f} -> {g_cmd:4.1f}"
    cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return img


def save_video(path, frames, fps=FPS):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()


class Mover:
    """The arm as a piece mover: IK hover, the pick policy, IK carry and place.

    Connects the robot and both cameras. Assumes the arm starts parked (PARK_RAW)."""

    def __init__(self, policy=POLICY, ckpt="last", jaw_roll=JAW_ROLL_DEG, max_s=15.0, retries=2, ensemble=False, release=15.0):
        self.jaw_roll, self.max_s, self.retries, self.release = jaw_roll, max_s, retries, release
        self.kin = Kinematics()
        self.calib = load_arm_calib()
        self.tool = self.calib.get("tool_offset", np.zeros(3))
        self.roll = calib_roll(self.calib)
        self.kin.set_roll(self.roll)
        self.taught = add_drop_zone(json.load(open(POINTS_PATH)), self.calib)
        self.n = self.calib["normal"]
        self.plans = {}
        self.policy, self.pre, self.post, self.relative = load_policy(policy, ckpt, ensemble)
        print(f"policy {policy} loaded on {DEVICE}; chunk {self.policy.config.chunk_size}, "
              f"n_action_steps {self.policy.config.n_action_steps}, ensemble {ensemble}, relative pose {self.relative}")
        LOG_DIR.mkdir(exist_ok=True)
        self.run_id = time.strftime("%m%d_%H%M")
        self.robot, self.top = connect_with_cameras()
        self.feats = dataset_features(self.robot)
        self.parked = True

    @property
    def squares(self):
        """Taught squares the arm can pick from and place on."""
        return {sq for sq in self.taught if not sq.startswith("_") and sq != DROP}

    def plan(self, sq):
        """(q_hover, height, tilt, roll) above a square, cached. Hover = centre of the OPEN jaw gap over the square."""
        if sq not in self.plans:
            self.plans[sq] = hover_plan(self.kin, self.calib, self.taught, held_tool(self.tool, GRIPPER_OPEN), self.roll, sq, self.jaw_roll)
        return self.plans[sq]

    def hover(self, sq):
        q_h = self.plan(sq)[0]
        if self.parked:
            unpark(self.robot, q_h)
            self.parked = False
        elif not self.glide(sq, held_tool(self.tool, GRIPPER_OPEN)):
            unpark(self.robot, q_h)

    def glide_path(self, sq, tool, hang=0.0, q_from=None):
        """Joint waypoints (URDF deg) from q_from (default: the arm now) to the hover above sq:
        tool point up to CLEAR_M + hang, straight across, down to the hover. None if a
        waypoint does not solve or the solution jumps branch."""
        kin, n = self.kin, self.n
        q_h, h_hover, _, roll = self.plan(sq)
        kin.set_roll(roll)
        q = read_joints(self.robot) if q_from is None else np.asarray(q_from, float)
        base = square_target(self.taught, self.calib, sq)
        p0 = kin.tool_pos(q, tool)
        h0 = float((p0 - base) @ n)
        H = max(CLEAR_M + hang, h0)
        flat = lambda p: p - float((p - base) @ n) * n  # projected onto the board plane
        a, b = flat(p0), flat(base)
        pts = [(a, h) for h in np.arange(h0, H, GLIDE_STEP_M)[1:]] + [(a, H)]
        # across: an arc about the pan axis (radius and angle interpolated), so a move along
        # the back rank sweeps around the base instead of cutting through the squares beside it
        ra, rb = np.hypot(*a[:2]), np.hypot(*b[:2])
        ta, tb = np.arctan2(a[1], a[0]), np.arctan2(b[1], b[0])
        tb = ta + (tb - ta + np.pi) % (2 * np.pi) - np.pi  # the short way round
        k = max(1, int(np.ceil(max(abs(rb - ra), abs(tb - ta) * max(ra, rb)) / GLIDE_STEP_M)))
        for i in range(1, k + 1):
            r, t = ra + (rb - ra) * i / k, ta + (tb - ta) * i / k
            pts.append((np.array([r * np.cos(t), r * np.sin(t), a[2] + (b[2] - a[2]) * i / k]), H))
        pts += [(b, h) for h in np.arange(H, h_hover, -GLIDE_STEP_M)[1:]]
        path = []
        for xy, h in pts:
            for hh in np.arange(max(h, h_hover), h_hover - 1e-6, -0.01):  # step down where the carry height does not solve
                q_w, err, _, bad = solve_point(kin, xy + hh * n, -n, tool, roll, q_warm=q)
                if not bad and err <= 0.005 and np.abs(q_w[:4] - q[:4]).max() <= GLIDE_MAX_JUMP_DEG:
                    break
            else:
                return None
            path.append(q_w)
            q = q_w
        if np.abs(q_h[:4] - q[:4]).max() > GLIDE_MAX_JUMP_DEG:
            return None
        return path + [q_h]

    def glide(self, sq, tool, hang=0.0, settle=True):
        """Stream glide_path() at GLIDE_SPEED and settle on the hover (settle=False: the hover
        is only passed through, as before a place). False if there is no path."""
        path = self.glide_path(sq, tool, hang)
        if path is None:
            print(f"   {sq}: no glide path; using the travel pose")
            return False
        q = read_joints(self.robot)
        for q_w in path:
            frames = max(2, int(round(np.linalg.norm(self.kin.tool_pos(q_w, tool) - self.kin.tool_pos(q, tool)) / GLIDE_SPEED * FPS)))
            for i in range(1, frames + 1):
                send_joints(self.robot, q + (q_w - q) * i / frames)
                time.sleep(1 / FPS)
            q = q_w
        if settle:
            move_to(self.robot, path[-1], duration_s=0.3)
        return True

    @contextmanager
    def filming(self, path):
        """Film wrist | top to an mp4 from a thread while the IK moves run. Cameras only: the servo bus stays with the caller."""
        frames, stop = [], threading.Event()

        def loop():
            t0 = time.perf_counter()
            while not stop.is_set():
                try:
                    obs = {"wrist": self.robot.cameras["wrist"].read_latest(max_age_ms=1000), "top": self.top.read()}
                    img = debug_frame(obs, 0.0, 0.0, 0.0, None, 0.0, f"t {time.perf_counter() - t0:4.1f}s")
                    frames.append(cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
                except TimeoutError:
                    pass
                stop.wait(1 / CARRY_FPS)

        th = threading.Thread(target=loop, daemon=True)
        th.start()
        try:
            yield
        finally:
            stop.set()
            th.join()
            save_video(path, frames, CARRY_FPS)

    def sideways(self, q, base):
        """Offset (m) of the open jaw gap's centre from the square centre along the pan swing, for joints q (URDF deg)."""
        r = np.array([base[0], base[1], 0.0])
        t = np.cross(self.n, r / np.linalg.norm(r))
        return float((self.kin.tool_pos(q, held_tool(self.tool, GRIPPER_OPEN)) - base) @ t / np.linalg.norm(t))

    def bound_sideways(self, act, q_raw, base):
        """Turn the commanded pan back until the jaws (the other joints where they are now, q_raw in lerobot
        deg) would be within SIDE_LIM_M of the square centre. Returns how far the command was over (m)."""
        q = np.asarray(q_raw, float) - OFFSET
        q[0] = act[f"{JOINTS[0]}.pos"] - OFFSET[0]
        first = 0.0
        for _ in range(2):  # the offset is nearly linear in the pan angle: two secant steps land within 0.1 mm
            s = self.sideways(q, base)
            over = s - float(np.clip(s, -SIDE_LIM_M, SIDE_LIM_M))
            if not over:
                break
            first = first or over
            q1 = q.copy()
            q1[0] += 0.5
            q[0] -= over / ((self.sideways(q1, base) - s) / 0.5)
        act[f"{JOINTS[0]}.pos"] = float(q[0] + OFFSET[0])
        return first

    def open_jaws(self):
        send_joints(self.robot, read_joints(self.robot), gripper=GRIPPER_OPEN)
        time.sleep(0.5)

    def park(self):
        if not self.parked:
            park(self.robot)
        self.parked = True

    def close(self):
        try:
            if not self.parked:
                self.park()
        finally:
            self.top.disconnect()
            self.robot.disconnect()

    def pick(self, src):
        """Hover above src and run the policy (with retries after a miss).

        Returns (q_close, g_end) when a piece is held at the hover, else None with the
        jaws open at the hover."""
        q_h, h_hover, _, roll_h = self.plan(src)
        base = square_target(self.taught, self.calib, src)
        height = lambda q_raw: float((self.kin.tool_pos(np.asarray(q_raw) - OFFSET, self.tool) - base) @ self.n)
        robot, top, policy = self.robot, self.top, self.policy
        self.hover(src)
        send_joints(robot, read_joints(robot), gripper=GRIPPER_OPEN)
        time.sleep(0.7)
        reason, q_close, g_end = "not run", None, None
        for attempt in range(1 + self.retries):
            if attempt:  # the same start repeats the same miss: retry from a slightly different one
                move_to(robot, jittered_hover(self.kin, self.calib, self.taught, self.tool, src, q_h, h_hover, roll_h, RETRY_SHIFT_MM), duration_s=1.0)
            policy.reset()
            print(f">> policy on {src}  (hover {h_hover * 100:.0f} cm, attempt {attempt + 1})")
            t0, closed_since, lowest, reason, saved, q_close = time.perf_counter(), None, np.inf, "timeout", False, None
            squeeze = False  # latched once the lift starts with the jaws closed
            bounded = 0.0  # the furthest the policy asked to go beyond SIDE_LIM_M
            q0 = None  # joint reading when this attempt starts: the origin for a relative-pose policy
            trace = []  # (t, 6 joint readings, 6 commanded) per frame, saved for offline comparison with the demos
            frames = []  # wrist | top side by side with the readings burnt in, saved as an mp4 per attempt
            while time.perf_counter() - t0 < self.max_s:
                tl = time.perf_counter()
                obs = robot.get_observation()
                obs["top"] = top.read()
                if not saved:  # what the policy saw first, for comparison with the training frames
                    cv2.imwrite(str(LOG_DIR / f"{self.run_id}_{src}_top.jpg"), cv2.cvtColor(obs["top"], cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(LOG_DIR / f"{self.run_id}_{src}_wrist.jpg"), cv2.cvtColor(obs["wrist"], cv2.COLOR_RGB2BGR))
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
                if q0 is None:
                    q0 = q_raw.copy()
                if self.relative:  # the policy sees and commands the arm joints as offsets from q0
                    act = infer(policy, self.pre, self.post, self.feats, {**obs, **{f"{j}.pos": obs[f"{j}.pos"] - v for j, v in zip(JOINTS, q0)}})
                    for j, v in zip(JOINTS, q0):
                        act[f"{j}.pos"] = float(act[f"{j}.pos"] + v)
                else:
                    act = infer(policy, self.pre, self.post, self.feats, obs)
                # the policy does not always command the jaws fully shut (demos hold at 2-11): empty jaws then
                # read above GRIPPER_EMPTY and pass as a hold, and a held piece is gripped lightly. Shut them.
                squeeze = squeeze or (closed_since is not None and tip_h > lowest + SQUEEZE_LIFT_M)
                if squeeze:
                    act["gripper.pos"] = 0.0
                over = self.bound_sideways(act, q_raw, base)
                bounded = max(bounded, abs(over))
                # the servos clamp to their calibrated range themselves; guard only against a wild output
                jump = {j: act[f"{j}.pos"] - obs[f"{j}.pos"] for j in JOINTS if abs(act[f"{j}.pos"] - obs[f"{j}.pos"]) > MAX_JUMP_DEG}
                if jump:
                    reason = f"command jumps too far from the current pose: { {j: round(v) for j, v in jump.items()} }"
                    break
                robot.send_action(act)
                trace.append([time.perf_counter() - t0, *q_raw, g, *(act[f"{j}.pos"] for j in JOINTS), act["gripper.pos"]])
                frames.append(debug_frame(obs, time.perf_counter() - t0, tip_h, height(np.array([act[f"{j}.pos"] for j in JOINTS])), g, act["gripper.pos"]))
                print(f"\r   t {time.perf_counter() - t0:4.1f}s  tip {tip_h * 1000:4.0f} mm  gripper {g:4.0f} -> {act['gripper.pos']:4.0f}   ", end="", flush=True)
                dt = time.perf_counter() - tl
                if dt < 1 / FPS:
                    time.sleep(1 / FPS - dt)
            if reason == "done":  # full squeeze before trusting the reading: empty jaws settle at about 5
                send_joints(robot, read_joints(robot), gripper=0.0)
                time.sleep(0.3)
            g_end = robot.get_observation()["gripper.pos"]
            if reason == "done" and g_end < GRIPPER_EMPTY:
                reason = "missed: gripper closed on nothing"
            np.save(LOG_DIR / f"{self.run_id}_{src}_{attempt + 1}_trace.npy", np.array(trace))
            save_video(LOG_DIR / f"{self.run_id}_{src}_{attempt + 1}.mp4", frames)
            print(f"\n   stopped: {reason} after {time.perf_counter() - t0:.1f} s; lowest tip {lowest * 1000:.0f} mm, gripper {g_end:.0f}, top-cam stalls {top.stalls}"
                  + (f", sideways bound held back {bounded * 1000:.0f} mm" if bounded else ""))
            if reason == "done" or not reason.startswith("missed"):
                break
            # missed: open where we are (don't drag a nudged piece), back to the hover, try again
            send_joints(robot, read_joints(robot), gripper=GRIPPER_OPEN)
            time.sleep(0.5)
            move_to(robot, q_h, duration_s=1.5)
        move_to(robot, q_h, duration_s=1.0)  # back up to the hover
        if reason != "done":
            if not GRIPPER_EMPTY <= g_end < GRIPPER_CLOSED:  # nothing in the jaws: leave them open
                send_joints(robot, read_joints(robot), gripper=GRIPPER_OPEN)
            return None
        return q_close, g_end

    def place(self, dst, src, q_close, g_end):
        """Carry the piece grasped on src (grasp pose q_close, gripper g_end) to dst and release it.

        dst == DROP: just let go above the drop zone. Returns False if the place IK fails
        (the piece is then still held at the hover above src)."""
        robot, kin = self.robot, self.kin
        q_h2, _, _, r2 = self.plan(dst)
        base2 = square_target(self.taught, self.calib, dst)
        n = self.n
        # place the piece's BASE (half a jaw gap from the touch point, hanging below the
        # fingers by however high it was grasped) PLACE_GAP_M above the square centre
        piece = held_piece_tool(kin, q_close, g_end, self.tool, square_target(self.taught, self.calib, src), n)
        kin.set_roll(r2)
        gap = DROP_GAP_M if dst == DROP else PLACE_GAP_M
        q_p, err, _, badp = solve_point(kin, base2 + gap * n, -n, piece, r2, q_warm=q_h2)
        if badp or err > 0.005:
            print(f"   {dst}: place IK failed (err {err * 1000:.0f} mm {badp})")
            return False
        shots = []  # wrist | top at each step of the release, saved as one image per placement
        with self.filming(LOG_DIR / f"{self.run_id}_{src}-{dst}_place.mp4"):
            if not self.glide(dst, piece, hang=float(np.linalg.norm(piece - self.tool))):
                unpark(robot, q_h2)  # carry via the raised travel pose, clear of the other pieces
            self.snap(shots, "carried")
            move_to(robot, q_p, duration_s=2.0)
            self.snap(shots, "lowered: base 2 mm up")
            if dst == DROP:
                send_joints(robot, read_joints(robot), gripper=g_end + self.release)
                time.sleep(0.5)
                move_to(robot, q_h2, duration_s=1.2)
                print(f"   dropped on the {DROP} zone")
                return True
            self.release_on(f"{src}-{dst}", dst, piece, g_end, q_h2, r2, base2, q_p, shots)
        print(f"   placed on {dst} (held at {g_end:.0f}, piece base {np.linalg.norm(piece - self.tool) * 1000:.0f} mm from the fingertips)")
        return True

    def snap(self, shots, label):
        obs = self.robot.get_observation()
        obs["top"] = self.top.read()
        shots.append(debug_frame(obs, len(shots), 0.0, 0.0, obs["gripper.pos"], 0.0, label))

    def release_on(self, tag, dst, piece, g_end, q_h2, r2, base2, q_p, shots):
        """Guided release from q_p (piece base PLACE_GAP_M above the square): press onto the mat
        (no error trimming, so a stalled servo is not pushed harder), crack the jaws so the piece
        settles on its base, open wide while still at the bottom (a jaw swinging away from a
        standing piece cannot move it; pads sliding up a widening piece can), rise, hover."""
        robot, kin, n = self.robot, self.kin, self.n
        q_t, err_t, _, bad_t = solve_point(kin, base2 - PLACE_PRESS_M * n, -n, piece, r2, q_warm=q_p)
        if not bad_t and err_t <= 0.005:
            q_now = read_joints(robot)
            for k in range(1, 16):
                send_joints(robot, q_now + (q_t - q_now) * k / 15)
                time.sleep(1 / FPS)
            time.sleep(0.3)
        self.snap(shots, "pressed 3 mm")
        send_joints(robot, read_joints(robot), gripper=g_end + RELEASE_CRACK)
        time.sleep(0.5)
        self.snap(shots, "cracked")
        send_joints(robot, read_joints(robot), gripper=g_end + self.release)
        time.sleep(0.5)
        self.snap(shots, "opened")
        # the fixed jaw is still against the piece and would drag it up: slide sideways until the
        # jaws, opening fully on the way, are centred on the piece, then rise straight up
        q_b = q_t if not bad_t and err_t <= 0.005 else q_p
        gt = held_tool(self.tool, GRIPPER_OPEN)
        q_c, err_c, _, bad_c = solve_point(kin, kin.tool_pos(q_b, held_tool(self.tool, g_end)), -n, gt, r2, q_warm=q_b)
        if not bad_c and err_c <= 0.005:
            q_now, g_now = read_joints(robot), g_end + self.release
            for k in range(1, 16):
                send_joints(robot, q_now + (q_c - q_now) * k / 15, gripper=g_now + (max(GRIPPER_OPEN, g_now) - g_now) * k / 15)
                time.sleep(1 / FPS)
            time.sleep(0.2)
        self.snap(shots, "jaws centred")
        if not self.glide(dst, gt, settle=False):
            move_to(robot, q_h2, duration_s=1.2)
        self.snap(shots, "back at hover")
        cv2.imwrite(str(LOG_DIR / f"{self.run_id}_{tag}_place.jpg"), np.vstack(shots))

    def move(self, src, dst):
        """Pick on src, place on dst (or DROP). Returns True when the piece was released on dst."""
        grasp = self.pick(src)
        if grasp is None:
            return False
        return self.place(dst, src, *grasp)


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

    m = Mover(a.policy, a.ckpt, a.jaw_roll, a.max_s, a.retries, a.ensemble, a.release)
    for sq in {sq for p in pairs for sq in p}:
        m.plan(sq)  # fail before touching the arm if a square is not taught / not solvable
    try:
        input(f"\n{len(pairs)} move(s): {' '.join(f'{s}->{d}' for s, d in pairs)}. Pieces in place? Hand on the switch? Enter to go... ")
        for src, dst in pairs:
            if a.no_place:
                print("   holding here; parking" if m.pick(src) else "   no piece held; parking")
                break
            if not m.move(src, dst):
                print("   move failed; parking")
                break
    finally:
        m.close()


if __name__ == "__main__":
    main()
