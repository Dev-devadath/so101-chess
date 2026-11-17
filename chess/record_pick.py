"""Record demos of the PICK skill for ACT: from the IK hover above a piece,
descend, grasp, lift back up. Everything else (travel, placing, parking) is
done by IK at play time, so that is also what this recorder does between
episodes.

Per episode:
  1. The arm is already hovering HOVER_M above the square (IK).
  2. Enter starts recording; the LEADER drives the follower in delta mode
     (follower = hover + leader movement on all 5 joints incl. roll, gripper
     follows the leader directly), so there is no jump. Descend, grasp, lift
     back to about hover height.
  3. Enter ends and saves the episode (r = discard and redo it, q = discard and quit).
  4. IK carries the piece to the next square, lowers it to the height it was
     grasped at and opens: the piece is now in place for the next episode and
     the arm hovers above it. Fix the piece if it fell, add/move neighbouring
     pieces, swap the piece type when prompted, then Enter.

Cameras: observation.images.wrist is the raw wrist camera (640x480).
observation.images.top is the overhead camera WARPED to the calibrated board
(board_calib.json -> TOP_SIZE x TOP_SIZE, a1 bottom-left), so the policy sees
the same board view whatever the tripod position. Run calibrate_board.py
first each session; the calibration used is copied into the dataset's meta/.

Usage:  .venv/bin/python chess/record_pick.py [--episodes 50] [--dataset so101_pick] [--resume]
                                               [--swap-every 8] [--place-h 25] [--back] [--jitter 6] [squares...]
--back: the piece goes back onto its own square after each episode (recording on a full
board, where the next square in the list is occupied).
--jitter MM: start each episode from the hover shifted by a random offset of up to MM mm in
the board plane, so the demo has to re-align over the piece. The policy only learns the
corrections it has seen: demos that start exactly above the piece teach a blind descent.
An episode only starts once the leader gripper is near GRIPPER_OPEN, and the fingertip
height is printed live while recording, so every demo starts and grasps alike
(see RECORDING_PLAN.md).
Squares default to all taught squares, shuffled. The dataset is a normal
LeRobotDataset (datasets/<name>), trainable with train_chess_act.sh / train_chess_gpu.sh.
"""

import argparse
import json
import random
import shutil
import sys
import time
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.datasets import LeRobotDataset, aggregate_pipeline_dataset_features, create_initial_features
from lerobot.processor import make_default_processors
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts

from arm import JOINTS, OFFSET, PORT, ROBOT_ID, Kinematics, load_arm_calib, move_to, park, read_gripper, read_joints, send_joints, unpark
from board import CALIB_PATH, WARP_SIZE, load_calib, warp_matrix
from cameras import OVERHEAD, WRIST, camera_index
from hover_squares import hover_plan, square_target
from reach_map import calib_roll, held_piece_tool, held_tool, solve_point
from teach_squares import LEADER_ID, LEADER_PORT, POINTS_PATH, read_line_nonblocking

FPS = 30
TASK = "pick up the chess piece"
SWAP_EVERY = 8  # episodes between "swap the piece type" prompts
TOP_SIZE = 480  # warped board image, px square
GRIPPER_OPEN = 40.0  # gripper reading the demos start with; the hover centres the open jaw gap on the square
GRIPPER_OPEN_TOL = 5.0  # the leader gripper must be this close to GRIPPER_OPEN before an episode starts
GRIPPER_CLOSED = 30.0  # follower reading below this = holding (with the pads, king/queen at the widest part read up to 24)
MAX_DELTA_DEG = 70.0  # leader further than this from where it was when recording started: hold the follower (the old demos peak at 65)
PLACE_GAP_M = 0.002  # piece base this far above the mat when the jaws open
MIN_FRAMES = 45  # an episode shorter than this (1.5 s; the shortest real demo is 98 frames) is an accidental Enter
JITTER_MAX_JUMP_DEG = 15.0  # a jittered hover further than this on any joint is an IK branch jump, not a shift
DATASETS_DIR = Path(__file__).parent.parent / "datasets"


class BoardCam:
    """Overhead camera read through lerobot's threaded camera, warped to the board."""

    def __init__(self):
        corners = load_calib()["corners"]
        s = TOP_SIZE / WARP_SIZE
        self.M = np.diag([s, s, 1.0]) @ warp_matrix(corners)
        self.cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=camera_index(OVERHEAD), fps=FPS, width=640, height=480))
        self.cam.connect()
        self.last, self.stalls = None, 0

    def read(self):
        """Warped board image. If the camera stalls (USB hiccup) the last frame is reused."""
        try:
            self.last = cv2.warpPerspective(self.cam.read_latest(max_age_ms=1000), self.M, (TOP_SIZE, TOP_SIZE))
        except TimeoutError:
            if self.last is None:
                raise
            self.stalls += 1
        return self.last

    def disconnect(self):
        self.cam.disconnect()


def connect_with_cameras():
    cams = {"wrist": OpenCVCameraConfig(index_or_path=camera_index(WRIST), fps=FPS, width=640, height=480)}
    robot = SO101Follower(SO101FollowerConfig(port=PORT, id=ROBOT_ID, cameras=cams))
    robot.connect()
    return robot, BoardCam()


def dataset_features(robot):
    """Same construction as lerobot-record (identity processors), plus the warped top view."""
    teleop_proc, _, obs_proc = make_default_processors()
    feats = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_proc, initial_features=create_initial_features(action=robot.action_features), use_videos=True
        ),
        aggregate_pipeline_dataset_features(
            pipeline=obs_proc, initial_features=create_initial_features(observation=robot.observation_features), use_videos=True
        ),
    )
    feats["observation.images.top"] = {**feats["observation.images.wrist"], "shape": (TOP_SIZE, TOP_SIZE, 3)}
    return feats


def open_dataset(name, robot, resume, root=None):
    root = root or DATASETS_DIR / name
    kw = dict(root=root, vcodec="h264", streaming_encoding=True, encoder_threads=2)
    if resume:
        ds = LeRobotDataset.resume(f"local/{name}", **kw)
    else:
        ds = LeRobotDataset.create(f"local/{name}", FPS, features=dataset_features(robot), robot_type=robot.name, use_videos=True, **kw)
    shutil.copy(CALIB_PATH, Path(root) / "meta" / f"board_calib_{date.today():%m%d}.json")  # which warp this session used
    return ds


def add_frame(dataset, obs, q_urdf, gripper):
    """One dataset frame: the robot observation (+ warped top view) and the commanded action in lerobot degrees."""
    sent = {f"{j}.pos": float(v) for j, v in zip(JOINTS, np.asarray(q_urdf) + OFFSET)}
    sent["gripper.pos"] = float(gripper)
    frame = {**build_dataset_frame(dataset.features, obs, OBS_STR), **build_dataset_frame(dataset.features, sent, ACTION), "task": TASK}
    dataset.add_frame(frame)


def record_episode(robot, top, leader, dataset, q_hover, tip_height):
    """Delta teleop from q_hover, one frame per tick.
    Returns (key that ended it, lowest tip height reached, joints (URDF deg) when the
    gripper last closed, or None if it is open)."""
    lead = leader.get_action()
    q_lead0 = np.array([lead[f"{j}.pos"] for j in JOINTS])
    print("   recording... Enter = save, r = redo, q = quit", flush=True)
    n, lowest, stalls, last_obs, q_close = 0, np.inf, 0, None, None
    top.stalls = 0
    while True:
        t0 = time.perf_counter()
        try:
            obs = robot.get_observation()
        except TimeoutError:  # wrist camera stalled: fresh joints, last image
            if last_obs is None:
                raise
            obs = {**last_obs, **{f"{m}.pos": v for m, v in robot.bus.sync_read("Present_Position").items()}}
            stalls += 1
        last_obs = obs
        obs["top"] = top.read()
        lead = leader.get_action()
        # unwrap: the leader's shoulder_lift reading crosses its encoder zero (+182 -> -176) mid-demo
        delta = (np.array([lead[f"{j}.pos"] for j in JOINTS]) - q_lead0 + 180.0) % 360.0 - 180.0
        q_urdf = np.array([obs[f"{j}.pos"] for j in JOINTS]) - OFFSET
        h = tip_height(q_urdf)
        if np.abs(delta).max() > MAX_DELTA_DEG:
            # the leader was picked up / put down mid-episode: copying that would swing the arm
            j = JOINTS[int(np.abs(delta).argmax())]
            print(f"\r   {j} {delta[np.abs(delta).argmax()]:+.0f} deg from the start pose: follower HELD. r = redo   ", end="", flush=True)
        else:
            q = q_hover + delta
            send_joints(robot, q, gripper=lead["gripper.pos"])
            add_frame(dataset, obs, q, lead["gripper.pos"])
            n += 1
            lowest = min(lowest, h)
            if obs["gripper.pos"] < GRIPPER_CLOSED:
                q_close = q_urdf if q_close is None else q_close  # pose at the grasp
            else:
                q_close = None
            print(f"\r   tip {h * 1000:4.0f} mm  gripper {obs['gripper.pos']:4.0f}   ", end="", flush=True)
        key = read_line_nonblocking()
        if key is not None:
            print(f"\n   {n} frames ({n / FPS:.1f} s), grasped {lowest * 1000:.0f} mm above the mat"
                  + (f"; camera stalls: wrist {stalls}, top {top.stalls}" if stalls or top.stalls else ""))
            if key not in ("q", "r") and (n < MIN_FRAMES or q_close is None):
                # Enter hit by accident: a one-frame episode breaks lerobot's episode index (the whole batch of 10 is lost)
                print(f"   {n} frames and {'no piece held' if q_close is None else 'a piece held'}: not a pick, discarding")
                key = "r"
            return key, lowest, q_close
        dt = time.perf_counter() - t0
        if dt < 1 / FPS:
            time.sleep(1 / FPS - dt)


def jittered_hover(kin, calib, taught, tool, sq, q_h, h, r, mm):
    """q_h (the hover above sq at height h, roll r) shifted by a random offset of up to mm in the board plane (uniform in a disc)."""
    n = calib["normal"]
    fd, rd = (np.asarray(calib[k]) / np.linalg.norm(calib[k]) for k in ("file_dir", "rank_dir"))
    ang, rad = random.uniform(0, 2 * np.pi), mm * np.sqrt(random.random())
    df, dr = rad * np.cos(ang), rad * np.sin(ang)
    target = square_target(taught, calib, sq) + h * n + (df * fd + dr * rd) / 1000
    q, err, _, bad = solve_point(kin, target, -n, held_tool(tool, GRIPPER_OPEN), r, q_warm=q_h)
    jump = np.abs(np.asarray(q)[:4] - np.asarray(q_h)[:4]).max()
    if bad or err > 0.005 or jump > JITTER_MAX_JUMP_DEG:  # a branch jump would snap the wrist, not shift it
        print(f"   jitter IK failed ({err * 1000:.0f} mm {bad}, jump {jump:.0f} deg); using the exact hover")
        return q_h
    print(f"   hover offset: {df:+.0f} mm along the files, {dr:+.0f} mm along the ranks")
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("squares", nargs="*")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--dataset", default="so101_pick")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--swap-every", type=int, default=SWAP_EVERY, help="prompt to change the piece every N episodes (0 = never, single-piece block)")
    ap.add_argument("--place-h", type=float, default=25.0, help="mm above the mat to open the gripper when placing (only until the first grasp height is known)")
    ap.add_argument("--release", type=float, default=15.0, help="open this much beyond the held reading when placing (keeps the jaws off the neighbours)")
    ap.add_argument("--back", action="store_true", help="set the piece back down on its own square after each episode (full board: every other square is occupied)")
    ap.add_argument("--jitter", type=float, default=0.0, help="mm: start each episode from the hover shifted by a random offset of up to this much in the board plane (0 = exact hover)")
    a = ap.parse_args()

    kin = Kinematics()
    calib = load_arm_calib()
    tool = calib.get("tool_offset", np.zeros(3))
    roll = calib_roll(calib)
    kin.set_roll(roll)
    taught = json.load(open(POINTS_PATH))
    n = calib["normal"]
    squares = a.squares or [s for s in taught if not s.startswith("_")]
    order = [squares[i % len(squares)] for i in range(a.episodes)]
    if not a.squares:
        random.shuffle(order)

    def plans(sq, piece=None):
        """(q_hover, q_place). piece = gripper-frame point to put PLACE_GAP_M above the square
        (the held piece's base); without one the fingertips go to --place-h."""
        # hover with the CENTRE of the open jaw gap over the square: the fixed jaw sits at the
        # tool point, so hovering the tool point there would bring it down on the piece
        q_h, _, _, r = hover_plan(kin, calib, taught, held_tool(tool, GRIPPER_OPEN), roll, sq)
        base = square_target(taught, calib, sq)
        target = base + (PLACE_GAP_M if piece is not None else a.place_h / 1000) * n
        q_p, err, _, bad = solve_point(kin, target, -n, tool if piece is None else piece, r, q_warm=q_h)
        if bad or err > 0.005:
            sys.exit(f"{sq}: place IK failed (err {err * 1000:.0f} mm {bad})")
        return q_h, q_p

    def tip_height_fn(sq):
        base = square_target(taught, calib, sq)
        return lambda q_urdf: float((kin.tool_pos(q_urdf, tool) - base) @ n)

    for sq in set(order):  # fail before touching the arm if any square cannot be planned
        plans(sq)
    print(f"{a.episodes} episodes -> datasets/{a.dataset}   order: {' '.join(order)}")

    robot, top = connect_with_cameras()
    leader = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id=LEADER_ID))
    leader.connect()
    dataset = open_dataset(a.dataset, robot, a.resume)
    saved = 0
    try:
        input(f"\nONE piece on {order[0]} (neighbours welcome), leader gripper OPEN. Enter when ready... ")
        q_h, _ = plans(order[0])
        unpark(robot, q_h)
        i = 0
        while i < len(order):
            sq = order[i]
            q_h, _ = plans(sq)
            q_start = q_h
            if a.jitter:
                _, h, _, r = hover_plan(kin, calib, taught, held_tool(tool, GRIPPER_OPEN), roll, sq)
                q_start = jittered_hover(kin, calib, taught, tool, sq, q_h, h, r, a.jitter)
                move_to(robot, q_start, duration_s=0.8)
            if a.swap_every and saved and saved % a.swap_every == 0:
                print(f"\n** {saved} saved: swap in a different piece type on {sq} (and rearrange neighbours) **")
            if input(f"\n[{saved + 1}/{a.episodes}] hovering over {sq}. Leader IN HAND and still, gripper open. Enter to record (q quit)... ").strip() == "q":
                break
            # every demo starts with the same jaw opening: wait for the leader gripper to be near GRIPPER_OPEN
            g_lead = leader.get_action()["gripper.pos"]
            if abs(g_lead - GRIPPER_OPEN) > GRIPPER_OPEN_TOL:
                print(f"   leader gripper reads {g_lead:.0f}: hold it at about {GRIPPER_OPEN:.0f} to start", flush=True)
                while abs(g_lead - GRIPPER_OPEN) > GRIPPER_OPEN_TOL:
                    time.sleep(0.2)
                    g_lead = leader.get_action()["gripper.pos"]
                    print(f"\r   leader gripper now {g_lead:5.1f} (need {GRIPPER_OPEN - GRIPPER_OPEN_TOL:.0f}-{GRIPPER_OPEN + GRIPPER_OPEN_TOL:.0f})   ", end="", flush=True)
                print()
            # match the follower gripper to the leader's before handing over, so it does not snap
            g_now, q_now = read_gripper(robot), read_joints(robot)
            for k in range(1, 11):
                send_joints(robot, q_now, gripper=g_now + (g_lead - g_now) * k / 10)
                time.sleep(0.05)
            key, grasp_h, q_close = record_episode(robot, top, leader, dataset, q_start, tip_height_fn(sq))
            if key == "q":
                dataset.clear_episode_buffer()
                break
            if key == "r":
                dataset.clear_episode_buffer()
                print("   discarded; put the piece back on the square and redo")
                move_to(robot, q_h, duration_s=1.5)
                continue
            dataset.save_episode()
            saved += 1
            i += 1
            # carry the piece to the next square via the raised travel pose and set its BASE
            # down PLACE_GAP_M above the mat (base located from the grasp pose + gripper reading)
            nxt = sq if a.back else (order[i] if i < len(order) else sq)
            g_held = read_gripper(robot)
            piece = held_piece_tool(kin, q_close, g_held, tool, square_target(taught, calib, sq), n) if q_close is not None else None
            q_h2, q_p2 = plans(nxt, piece)
            move_to(robot, q_h, duration_s=1.0)  # back to the exact hover first
            unpark(robot, q_h2)
            move_to(robot, q_p2, duration_s=1.5)
            send_joints(robot, read_joints(robot), gripper=g_held + a.release)
            time.sleep(0.6)
            move_to(robot, q_h2, duration_s=1.2)
            if a.back and i < len(order) and order[i] != sq:
                unpark(robot, plans(order[i])[0])  # the piece stayed here: travel on to the next square's hover
    finally:
        try:
            park(robot)
        finally:
            dataset.finalize()
            leader.disconnect()
            top.disconnect()
            robot.disconnect()
    print(f"\n{saved} episodes saved this run; datasets/{a.dataset} now has {dataset.num_episodes} episodes")


if __name__ == "__main__":
    main()
