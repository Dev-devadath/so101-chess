"""Which squares can the arm hover over with the gripper pointing down?

Dry run (no arm needed):   .venv/bin/python chess/reach_map.py
Visit each reachable square (CLEAR THE BOARD FIRST):
                           .venv/bin/python chess/reach_map.py --move          (Enter between squares)
                           .venv/bin/python chess/reach_map.py --move --auto   (no pauses; saves an
                           overhead snapshot per square to chess/hover_snaps/)

Map legend:  o reachable   x IK position error > tolerance   L joint limit hit
             t reachable but gripper tilted more than TILT_TOL from vertical
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from arm import AXIS_WEIGHT, JOINTS, Kinematics, connect, load_arm_calib, move_to, read_joints, square_xyz, within_limits
from board import FILES, draw_grid, load_calib, open_overhead, warp_board

SNAP_DIR = Path(__file__).parent / "hover_snaps"

HOVER_M = 0.08  # height above the board surface
# Jaws open along the board DIAGONAL (neighbours 69 mm away instead of 49 mm): hover/grasp
# poses roll the wrist by this much from the calibration roll. hover_plan() tries +/-.
JAW_ROLL_DEG = 45.0
# Moving-jaw geometry in gripper_frame_link, from the URDF: the jaw pivots about the
# frame's y axis and its tip swings along x. The calibrated tool point (the jaws' touch
# point, x = -33 mm) is where the tip sits at JAW_CLOSED_DEG, i.e. a reading of ~2.
JAW_PIVOT = np.array([-0.0281, 0.0, -0.0747])
JAW_TIP_URDF0 = np.array([-0.0189, 0.0, 0.0073])
JAW_CLOSED_DEG = 10.0
JAW_CLOSED_READING = 2.0
JAW_DEG_PER_UNIT = 1068 * 360 / 4096 / 100  # follower gripper calibration: 1068 ticks over the 0-100 scale
POS_TOL_M = 0.005
TILT_TOL_DEG = 30.0
# IK seeds (lift, elbow, wrist): from folded-near to stretched-far. The local
# solver gets stuck in joint limits from a single seed, so try all and keep the best.
SEEDS = [(-60, 60, 60), (-20, 30, 80), (0, 0, 90), (30, -30, 90), (60, -70, 95), (80, -90, 80)]


def hover_point(calib, file_idx, rank_idx):
    return square_xyz(calib, file_idx, rank_idx) + HOVER_M * calib["normal"]


def calib_roll(calib):
    """Wrist roll (URDF deg) used during touch calibration; the tool offset is valid for it."""
    from arm import OFFSET

    rolls = [q[4] for _, q in calib.get("joints_raw", [])]
    return float(np.mean(rolls)) - OFFSET[4] if rolls else 0.0


def jaw_tip(reading):
    """Moving-jaw tip in gripper_frame_link (m) for a gripper reading on the 0-100 scale."""
    th = np.deg2rad(JAW_CLOSED_DEG + JAW_DEG_PER_UNIT * (reading - JAW_CLOSED_READING))
    x, z = JAW_TIP_URDF0[0] - JAW_PIVOT[0], JAW_TIP_URDF0[2] - JAW_PIVOT[2]
    return JAW_PIVOT + np.array([x * np.cos(th) - z * np.sin(th), 0.0, x * np.sin(th) + z * np.cos(th)])


def held_tool(tool, reading):
    """Tool vector for the CENTRE of a piece held with the gripper reading this value.

    The piece is pushed against the fixed jaw, so its centre sits half the jaw gap
    from the touch point along the jaw axis. Use this as the tool for a place solve."""
    gap = jaw_tip(reading)[0] - jaw_tip(JAW_CLOSED_READING)[0]
    return np.asarray(tool, float) + np.array([0.5 * gap, 0.0, 0.0])


def held_piece_tool(kin, q_close, reading, tool, surface, n):
    """Gripper-frame vector (m) of the BASE of the piece being held, for a place solve.

    At q_close (URDF deg, the pose when the jaws closed) the piece stood on the mat:
    its centre is held_tool() in the world, its base is that dropped onto the mat
    along n. Expressed in the gripper frame it is rigid while the piece is held,
    whatever the tilt at either end, so solve_point(..., tool=this, ...) puts the
    piece's base on the target."""
    T = kin.fk(q_close)
    centre = (T @ np.r_[held_tool(tool, reading), 1.0])[:3]
    base = centre - float((centre - surface) @ n) * n
    return (np.linalg.inv(T) @ np.r_[base, 1.0])[:3]


def solve_point(kin, pos, point_dir, tool, roll, q_warm=None):
    """Best IK over all seeds, pan pre-aimed at the target. Returns (q, err, tilt, bad).

    q_warm: an extra full seed, e.g. the solution for a nearby point."""
    pan = -np.degrees(np.arctan2(pos[1], pos[0]))  # +pan swings towards -y
    seeds = [np.array([pan, lift, elbow, wrist, roll]) for lift, elbow, wrist in SEEDS]
    if q_warm is not None:
        seeds.insert(0, np.asarray(q_warm, float))
    def best_over_seeds():
        best = None
        for seed in seeds:
            q, err, tilt = kin.ik_tool(seed, pos, point_dir, tool)
            bad = within_limits(q)
            cost = err + (0.1 if bad else 0) + 1e-4 * tilt
            if best is None or cost < best[0]:
                best = (cost, q, err, tilt, bad)
        return best

    best = best_over_seeds()
    if best[2] > POS_TOL_M or best[4]:
        # Where pointing straight down is impossible (far squares) the axis task makes
        # placo's step overshoot into the joint limits. Solve position-only and accept the tilt.
        kin.axis_task.configure("gripper_axis", "soft", 0.0)
        try:
            alt = best_over_seeds()
        finally:
            kin.axis_task.configure("gripper_axis", "soft", AXIS_WEIGHT)
        if alt[0] < best[0]:
            best = alt
    return best[1:]


def solve_all(kin, calib):
    tool = calib.get("tool_offset", np.zeros(3))
    roll = calib_roll(calib)
    kin.set_roll(roll)
    print(f"tool offset {np.round(tool * 1000)} mm, wrist roll held at {roll:.0f} deg")
    results = {}
    for f in range(8):
        for r in range(8):
            q, err, tilt, bad = solve_point(kin, hover_point(calib, f, r), -calib["normal"], tool, roll)
            if bad:
                status = "L"
            elif err > POS_TOL_M:
                status = "x"
            elif tilt > TILT_TOL_DEG:
                status = "t"
            else:
                status = "o"
            results[(f, r)] = (status, q, err, tilt, bad)
    return results


def print_map(results):
    for r in range(7, -1, -1):
        print(f"{r + 1}  " + " ".join(results[(f, r)][0] for f in range(8)))
    print("   " + " ".join(FILES))
    n_ok = sum(1 for v in results.values() if v[0] == "o")
    print(f"\n{n_ok}/64 reachable at {HOVER_M * 100:.0f} cm hover")
    for (f, r), (status, q, err, tilt, bad) in sorted(results.items()):
        if status != "o":
            print(f"  {FILES[f]}{r + 1}: err {err * 1000:.0f} mm  tilt {tilt:.0f} deg  {' '.join(bad)}")


def main():
    kin = Kinematics()
    calib = load_arm_calib()
    results = solve_all(kin, calib)
    print_map(results)

    if "--move" not in sys.argv:
        return

    auto = "--auto" in sys.argv
    cap = None
    if auto:
        cam = load_calib()
        cap = open_overhead()
        SNAP_DIR.mkdir(exist_ok=True)
    input("\nboard clear? arm will visit every reachable square. Enter to start, Ctrl-C to stop... ")
    robot = connect()
    try:
        for r in range(8):
            for f in range(8):
                status, q, err, tilt, _ = results[(f, r)]
                if status != "o":
                    continue
                name = f"{FILES[f]}{r + 1}"
                print(f"{name}: {np.round(q, 1)}")
                track_err = move_to(robot, q, duration_s=2.0)
                actual = kin.tool_pos(read_joints(robot), calib.get("tool_offset", np.zeros(3)))
                target = hover_point(calib, f, r)
                print(
                    f"   tracking error per joint {np.round(track_err, 1)} deg"
                    f" -> tool {np.linalg.norm(actual - target) * 1000:.0f} mm from target"
                )
                if auto:
                    time.sleep(1.0)  # settle
                    for _ in range(5):  # flush buffered frames
                        cap.read()
                    ok, frame = cap.read()
                    if ok:
                        cv2.imwrite(str(SNAP_DIR / f"{name}.png"), draw_grid(warp_board(frame, cam["corners"])))
                else:
                    input("   Enter for next square... ")
    finally:
        robot.disconnect()
        if cap is not None:
            cap.release()


if __name__ == "__main__":
    main()
