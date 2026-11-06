"""Measure where a placed piece actually landed, with the overhead camera.

Shows the live warped board (calibrate_board.py first). Click the CENTRE OF THE
PIECE'S BASE: prints the square it is on and the offset of that point from the
target square's centre, in mm along the files and ranks. Several clicks are fine.

Usage:  .venv/bin/python chess/place_check.py g4        offsets from g4's centre
        .venv/bin/python chess/place_check.py           offsets from the nearest square's centre
Keys:   q quit
"""

import sys

import cv2

from board import FILES, WARP_SIZE, draw_grid, load_calib, open_overhead, warp_board

SQUARE_MM = 49.0
SQ = WARP_SIZE // 8
MM_PER_PX = SQUARE_MM / SQ


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else None
    corners = load_calib()["corners"]
    cap = open_overhead()
    clicks = []

    def on_click(event, x, y, *_):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        f, r = x // SQ, 7 - y // SQ
        tf, tr = (FILES.index(target[0]), int(target[1]) - 1) if target else (f, r)
        df = (x - (tf + 0.5) * SQ) * MM_PER_PX  # + towards the h file
        dr = ((7 - tr + 0.5) * SQ - y) * MM_PER_PX  # + towards rank 8
        clicks.append((x, y))
        print(f"on {FILES[f]}{r + 1}: {df:+5.1f} mm along the files, {dr:+5.1f} mm along the ranks from {FILES[tf]}{tr + 1}'s centre  (|{(df**2 + dr**2) ** 0.5:.1f}| mm)")

    cv2.namedWindow("place check")
    cv2.setMouseCallback("place check", on_click)
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        img = draw_grid(warp_board(frame, corners))
        for x, y in clicks:
            cv2.drawMarker(img, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
        cv2.imshow("place check", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
