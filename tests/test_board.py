"""board.piece_base on synthetic frames: a base disc with a body leaning away from the nadir."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "chess"))
from board import SQUARE_MM, WARP_SIZE, piece_base  # noqa: E402

SIZE = 480
SQ = SIZE / 8
NADIR = [437.0, 1384.0]  # warped px at WARP_SIZE, below the rank-1 edge like the real camera


def frames(sq, d_file_mm, d_rank_mm, radius_mm=13.0, colour=(40, 40, 40)):
    before = np.full((SIZE, SIZE, 3), 170, np.uint8)
    after = before.copy()
    f, r = "abcdefgh".index(sq[0]), int(sq[1]) - 1
    px = SQ / SQUARE_MM
    cx, cy = f * SQ + SQ / 2 + d_file_mm * px, (7 - r) * SQ + SQ / 2 - d_rank_mm * px
    d = np.array([cx, cy]) - np.array(NADIR) * SIZE / WARP_SIZE
    d /= np.linalg.norm(d)
    # body: a thinner bar from the base centre away from the nadir (the lean), then the base disc
    top = np.array([cx, cy]) + d * 2.5 * radius_mm * px
    cv2.line(after, (int(cx), int(cy)), (int(top[0]), int(top[1])), colour, int(radius_mm * px))
    cv2.circle(after, (int(round(cx)), int(round(cy))), int(round(radius_mm * px)), colour, -1)
    return before, after


class PieceBase(unittest.TestCase):
    def check(self, sq, df, dr, radius=13.0):
        got = piece_base(*frames(sq, df, dr, radius), sq, NADIR)
        self.assertIsNotNone(got, sq)
        self.assertAlmostEqual(got[0], df, delta=1.5, msg=f"{sq} file")
        self.assertAlmostEqual(got[1], dr, delta=1.5, msg=f"{sq} rank")
        self.assertAlmostEqual(got[2], radius, delta=2.0, msg=f"{sq} radius")

    def test_centred(self):
        for sq in ("a1", "e4", "h8", "d2"):
            self.check(sq, 0.0, 0.0)

    def test_offsets(self):
        self.check("e4", 6.0, -4.0)
        self.check("c3", -8.0, 7.0)
        self.check("g6", 10.0, 10.0, radius=18.0)

    def test_empty(self):
        before, _ = frames("e4", 0, 0)
        self.assertIsNone(piece_base(before, before, "e4", NADIR))

    def test_exclude_origin(self):
        before, after = frames("e4", 3.0, 0.0)
        # the piece that left the adjacent square shows as a change there; it must be ignored
        cv2.circle(before, (int(3 * SQ + SQ / 2), int(4 * SQ + SQ / 2)), 10, (40, 40, 40), -1)  # d4 in before only
        got = piece_base(before, after, "e4", NADIR, exclude="d4")
        self.assertAlmostEqual(got[0], 3.0, delta=1.5)


if __name__ == "__main__":
    unittest.main()
