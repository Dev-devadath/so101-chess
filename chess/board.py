"""Board calibration: warp the overhead camera image to a top-down 8x8 view.

Calibration file (board_calib.json) holds the pixel coords of the four outer
corners of the 64-square playing area in the order a1, h1, h8, a8.
The warped image is a standard diagram view: a1 bottom-left, h8 top-right.
"""

import json
from pathlib import Path

import cv2
import numpy as np

CALIB_PATH = Path(__file__).parent / "board_calib.json"
WARP_SIZE = 800  # px, square
SQ = WARP_SIZE // 8

FILES = "abcdefgh"


def load_calib(path=CALIB_PATH):
    with open(path) as f:
        return json.load(f)


def open_overhead():
    """The overhead camera, opened by name (indices reshuffle on macOS)."""
    from cameras import OVERHEAD, open_camera

    return open_camera(OVERHEAD)


def save_calib(corners, path=CALIB_PATH):
    data = {
        "corners": [[float(x), float(y)] for x, y in corners],
        "corner_order": ["a1", "h1", "h8", "a8"],
        "warp_size": WARP_SIZE,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return data


def warp_matrix(corners):
    src = np.float32(corners)
    # a1 bottom-left, h1 bottom-right, h8 top-right, a8 top-left
    dst = np.float32([[0, WARP_SIZE], [WARP_SIZE, WARP_SIZE], [WARP_SIZE, 0], [0, 0]])
    return cv2.getPerspectiveTransform(src, dst)


def warp_board(frame, corners):
    return cv2.warpPerspective(frame, warp_matrix(corners), (WARP_SIZE, WARP_SIZE))


def square_roi(warped, file_idx, rank_idx):
    """Pixel crop of a square. file_idx 0..7 = a..h, rank_idx 0..7 = 1..8."""
    x0 = file_idx * SQ
    y0 = (7 - rank_idx) * SQ
    return warped[y0 : y0 + SQ, x0 : x0 + SQ]


# Pieces lean away from the camera nadir (parallax), so a neighbour's top can
# spill into a square. The base never leaves its square, so sample a patch
# shifted away from the nadir: it holds the base and dodges the spill.
PATCH_SHIFT = 0.22  # fraction of a square
PATCH_HALF = 0.22


def square_patch_rect(file_idx, rank_idx, nadir):
    """(x0, y0, x1, y1) of the clean sample patch for a square, in warped px."""
    cx = file_idx * SQ + SQ / 2
    cy = (7 - rank_idx) * SQ + SQ / 2
    d = np.array([cx, cy]) - np.array(nadir, dtype=float)
    d /= np.linalg.norm(d) + 1e-9
    px, py = cx + PATCH_SHIFT * SQ * d[0], cy + PATCH_SHIFT * SQ * d[1]
    h = PATCH_HALF * SQ
    return int(px - h), int(py - h), int(px + h), int(py + h)


def square_patch(warped, file_idx, rank_idx, nadir):
    x0, y0, x1, y1 = square_patch_rect(file_idx, rank_idx, nadir)
    return warped[y0:y1, x0:x1]


def nadir_from_leans(leans):
    """Least-squares intersection of base->top lines. leans = [((bx,by),(tx,ty)), ...]."""
    A = np.zeros((2, 2))
    b = np.zeros(2)
    for base, top in leans:
        base, top = np.array(base, float), np.array(top, float)
        d = base - top
        d /= np.linalg.norm(d)
        P = np.eye(2) - np.outer(d, d)  # projector onto the line's normal
        A += P
        b += P @ base
    return np.linalg.solve(A, b)


def draw_patches(warped, nadir):
    out = warped.copy()
    for f in range(8):
        for r in range(8):
            x0, y0, x1, y1 = square_patch_rect(f, r, nadir)
            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 1)
    cv2.circle(out, tuple(int(v) for v in nadir), 8, (255, 0, 0), -1)
    return out


def draw_grid(warped):
    out = warped.copy()
    for i in range(1, 8):
        cv2.line(out, (i * SQ, 0), (i * SQ, WARP_SIZE), (0, 0, 255), 1)
        cv2.line(out, (0, i * SQ), (WARP_SIZE, i * SQ), (0, 0, 255), 1)
    for f in range(8):
        for r in range(8):
            cv2.putText(
                out,
                f"{FILES[f]}{r + 1}",
                (f * SQ + 4, (7 - r) * SQ + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
            )
    return out


# ---- where a piece's base is, from a before/after pair --------------------------------

SQUARE_MM = 49.0
BASE_PIXEL_DIFF = 10.0  # per-pixel Lab distance counted as the moved piece (frames minutes apart, same lighting)
BASE_MARGIN = 0.35  # squares: the search region extends this far past the square on every side
BASE_MIN_PX = 60  # fewer changed pixels than this: no piece found
BASE_R_MM = (9.0, 20.0)  # plausible base radius (pawn ~13 mm, king ~19 mm)


def piece_base(before, after, sq, nadir, exclude=None):
    """Where the piece that appeared on sq stands, from two warped frames (any size, same
    size) taken before and after the move. Returns (d_file_mm, d_rank_mm, radius_mm,
    changed_px) of the base centre relative to the square centre, or None.

    Pieces lean away from the camera nadir in the warped image, so the edge of the
    changed blob nearest the nadir is the base rim; the base centre is one radius
    further along, and its sideways position is the middle of the rim's width.
    exclude: a square whose changes are ignored (the move's origin, when adjacent)."""
    size = after.shape[0]
    sq_px = size / 8
    f, r = FILES.index(sq[0]), int(sq[1]) - 1
    cx, cy = f * sq_px + sq_px / 2, (7 - r) * sq_px + sq_px / 2
    lab = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    mask = (np.linalg.norm(lab(after) - lab(before), axis=2) > BASE_PIXEL_DIFF).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    roi = np.zeros_like(mask)
    m = (0.5 + BASE_MARGIN) * sq_px
    roi[max(int(cy - m), 0):int(cy + m), max(int(cx - m), 0):int(cx + m)] = 1
    if exclude:
        fe, re_ = FILES.index(exclude[0]), int(exclude[1]) - 1
        roi[int((7 - re_) * sq_px):int((8 - re_) * sq_px), int(fe * sq_px):int((fe + 1) * sq_px)] = 0
    mask &= roi
    ys, xs = np.nonzero(mask)
    if len(xs) < BASE_MIN_PX:
        return None
    d = np.array([cx, cy]) - np.asarray(nadir, float) * size / WARP_SIZE
    d /= np.linalg.norm(d)
    perp = np.array([-d[1], d[0]])
    p = np.stack([xs - cx, ys - cy], 1)
    s, t = p @ d, p @ perp
    s_near = np.percentile(s, 1)
    mm_per_px = SQUARE_MM / sq_px
    radius = BASE_R_MM[0] / mm_per_px
    for _ in range(3):  # the rim's width within one radius of the near edge is the diameter
        band = s < s_near + 1.2 * radius
        t_lo, t_hi = np.percentile(t[band], 0.5), np.percentile(t[band], 99.5)
        radius = np.clip((t_hi - t_lo) / 2, BASE_R_MM[0] / mm_per_px, BASE_R_MM[1] / mm_per_px)
    c = (s_near + radius) * d + (t_lo + t_hi) / 2 * perp
    return c[0] * mm_per_px, -c[1] * mm_per_px, radius * mm_per_px, len(xs)
