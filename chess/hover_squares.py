"""Visit the hover pose above taught squares, one Enter per square.

This is the IK half of the hybrid plan: the arm is brought to HOVER_M above
the square and the learned policy does the descend/grab/lift from there.
The gripper is not touched. Starts with unpark(), ends with park().

Usage:  .venv/bin/python chess/hover_squares.py d2 d4 c5     visit these in order
        .venv/bin/python chess/hover_squares.py               all taught squares
        .venv/bin/python chess/hover_squares.py g4 --height 10   fingertips 10 mm up (BOARD EMPTY: model check)
Keys (then Enter):  <Enter> next square   q park and quit
"""

import json
import sys

import numpy as np

from arm import Kinematics, connect, load_arm_calib, move_to, park, unpark
from reach_map import HOVER_M, JAW_ROLL_DEG, calib_roll, solve_point
from teach_squares import POINTS_PATH, surface_point


def square_target(taught, calib, sq):
    """Surface centre of a taught square for poses at the JAW roll.

    The squares were touched at the calibration roll; at the diagonal-jaw roll the arm
    lands 1-3 cm off, square-dependent. The error measured by the overhead camera
    (calibrate_squares_cam.py) is stored per square as taught[sq]["jaw_xy_mm"]
    = [along the files, along the ranks], falling back to the board-wide
    _jaw_roll_xy_mm (place_check.py), and subtracted here."""
    corr = taught[sq].get("jaw_xy_mm", taught.get("_jaw_roll_xy_mm", [0.0, 0.0]))
    fd, rd = (np.asarray(calib[k]) / np.linalg.norm(calib[k]) for k in ("file_dir", "rank_dir"))
    return surface_point(taught[sq]["xyz"], taught, calib) - (corr[0] * fd + corr[1] * rd) / 1000


def hover_plan(kin, calib, taught, tool, roll, sq, jaw_roll=JAW_ROLL_DEG, height=HOVER_M):
    """(q, height_m, tilt_deg, roll_deg) for hovering above a taught square, jaws diagonal.

    roll is the calibration roll; the wrist is rolled +jaw_roll (or -jaw_roll where
    only that solves, e.g. e1). jaw_roll=0 gives the old sideways jaws (datasets
    recorded before 2026-11-04). Squares next to the base cannot be hovered at full
    height, so the height steps down. kin is left set to the roll returned: use it
    for any further solve_point() at that square."""
    if sq not in taught:
        sys.exit(f"{sq} is not taught (run teach_squares.py {sq})")
    n = calib["normal"]
    base = square_target(taught, calib, sq)
    heights = np.arange(height, 0.039, -0.01) if height > 0.04 else [height]
    for r in (roll + jaw_roll, roll - jaw_roll):
        kin.set_roll(r)
        for h in heights:
            q, err, tilt, bad = solve_point(kin, base + h * n, -n, tool, r)
            if not bad and err <= 0.005:
                return q, h, tilt, r
    sys.exit(f"{sq}: IK failed at both jaw rolls (err {err * 1000:.0f} mm {bad})")


def main():
    kin = Kinematics()
    calib = load_arm_calib()
    tool = calib.get("tool_offset", np.zeros(3))
    roll = calib_roll(calib)
    kin.set_roll(roll)
    taught = json.load(open(POINTS_PATH))
    args = sys.argv[1:]
    height = HOVER_M
    if "--height" in args:  # mm; e.g. 10 to check where the closed fingertips sit over a square
        i = args.index("--height")
        height = float(args[i + 1]) / 1000
        del args[i:i + 2]
    squares = args or [s for s in taught if not s.startswith("_")]

    plans = []
    for sq in squares:
        q, h, tilt, _ = hover_plan(kin, calib, taught, tool, roll, sq, height=height)
        plans.append((sq, q, tilt, h))

    input(f"hover over {len(plans)} squares at {height * 100:.1f} cm. Enter to start... ")
    robot = connect()
    try:
        for i, (sq, q, tilt, h) in enumerate(plans):
            if i == 0:
                unpark(robot, q)
            else:
                move_to(robot, q, duration_s=2.0)
            if input(f"   {sq} ({h * 100:.0f} cm up, tilt {tilt:.0f} deg)  [Enter next, q quit] ").strip() == "q":
                break
        park(robot)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
