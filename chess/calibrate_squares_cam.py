"""Per-square position correction for the jaw-roll poses, measured with the overhead camera.

BOARD EMPTY, board calibrated (calibrate_board.py). The arm visits each square with
the closed fingertips 10 mm above where the model puts the square's centre. In the
camera window (the 3x3 squares around the target, magnified), CLICK THE FINGERTIPS:
the offset from the square's centre is added to that square's correction
(taught[sq]["jaw_xy_mm"] in square_points.json, saved at once). The arm then revisits
the square until the residual is under TOL_MM (at most MAX_VISITS), so one wrong click
is caught on the next visit; a square that keeps bouncing means the arm does not repeat
there. hover_plan()/square_target() apply the correction to every hover and place.

Usage:  .venv/bin/python chess/calibrate_squares_cam.py            all taught squares
        .venv/bin/python chess/calibrate_squares_cam.py g4 h5      just these
Window keys:  click = record and next     s = skip this square     q = park and quit
"""

import json
import sys

import cv2
import numpy as np

from arm import Kinematics, connect, load_arm_calib, move_to, park, unpark
from board import FILES, WARP_SIZE, draw_grid, load_calib, open_overhead, warp_board
from hover_squares import hover_plan
from reach_map import HOVER_M, calib_roll
from teach_squares import POINTS_PATH

CHECK_M = 0.01
SQUARE_MM = 49.0
SQ = WARP_SIZE // 8
MM_PER_PX = SQUARE_MM / SQ
TOL_MM = 3.0  # revisit the square until the fingertips are this close to its centre
MAX_VISITS = 3
ZOOM = 2  # the window shows the 3x3 squares around the target, magnified


def main():
    kin = Kinematics()
    calib = load_arm_calib()
    tool = calib.get("tool_offset", np.zeros(3))
    roll = calib_roll(calib)
    taught = json.load(open(POINTS_PATH))
    squares = sys.argv[1:] or [s for s in taught if not s.startswith("_")]
    corners = load_calib()["corners"]

    input(f"BOARD EMPTY? will visit {len(squares)} squares at {CHECK_M * 1000:.0f} mm. Enter to start... ")
    cap = open_overhead()
    robot = connect()
    click = []
    cv2.namedWindow("fingertips")
    cv2.setMouseCallback("fingertips", lambda ev, x, y, *_: click.append((x, y)) if ev == cv2.EVENT_LBUTTONDOWN else None)
    first = True
    try:
        for sq in squares:
            f, r = FILES.index(sq[0]), int(sq[1]) - 1
            cx, cy = int((f + 0.5) * SQ), int((7 - r + 0.5) * SQ)
            x0, y0 = min(max(cx - 3 * SQ // 2, 0), WARP_SIZE - 3 * SQ), min(max(cy - 3 * SQ // 2, 0), WARP_SIZE - 3 * SQ)
            for visit in range(MAX_VISITS):
                # corrections already stored are applied by hover_plan, so what we measure is the residual
                q_hover, *_ = hover_plan(kin, calib, taught, tool, roll, sq)
                q_low, _, tilt, _ = hover_plan(kin, calib, taught, tool, roll, sq, height=CHECK_M)
                if first:
                    unpark(robot, q_hover)
                    first = False
                else:
                    move_to(robot, q_hover, duration_s=2.0)
                move_to(robot, q_low, duration_s=1.5)
                old = taught[sq].get("jaw_xy_mm", taught.get("_jaw_roll_xy_mm", [0.0, 0.0]))
                print(f"{sq} visit {visit + 1}: tilt {tilt:.0f} deg, stored correction {old[0]:+.1f}/{old[1]:+.1f} mm. Click the fingertips (s skip, q quit)")
                click.clear()
                skipped = False
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    img = draw_grid(warp_board(frame, corners))
                    cv2.drawMarker(img, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
                    crop = img[y0:y0 + 3 * SQ, x0:x0 + 3 * SQ]
                    cv2.imshow("fingertips", cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_LINEAR))
                    key = cv2.waitKey(1) & 0xFF
                    if click:
                        x, y = x0 + click[0][0] / ZOOM, y0 + click[0][1] / ZOOM
                        df, dr = (x - cx) * MM_PER_PX, (cy - y) * MM_PER_PX
                        taught[sq]["jaw_xy_mm"] = [round(old[0] + df, 1), round(old[1] + dr, 1)]
                        json.dump(taught, open(POINTS_PATH, "w"), indent=1)
                        print(f"   fingertips {df:+.1f}/{dr:+.1f} mm off -> correction now {taught[sq]['jaw_xy_mm'][0]:+.1f}/{taught[sq]['jaw_xy_mm'][1]:+.1f} mm")
                        break
                    if key == ord("s"):
                        print("   skipped")
                        skipped = True
                        break
                    if key == ord("q"):
                        raise KeyboardInterrupt
                move_to(robot, q_hover, duration_s=1.2)
                if skipped or max(abs(df), abs(dr)) <= TOL_MM:
                    break
            else:
                print(f"   {sq} did not settle in {MAX_VISITS} visits: arm not repeating here, or inconsistent clicks")
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        try:
            park(robot)
        finally:
            robot.disconnect()
            cap.release()
            cv2.destroyAllWindows()
    done = [s for s in squares if "jaw_xy_mm" in taught[s]]
    print(f"{len(done)} squares have a per-square correction; rerun to measure the residual")


if __name__ == "__main__":
    main()
