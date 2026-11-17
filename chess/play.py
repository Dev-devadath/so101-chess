"""Play a game against the arm. The arm is White (rank 1 is at its base), you are Black.

Each turn:
  you      move a piece on the board and press Enter. The overhead camera reads
           empty/white/black per square; the one legal move that explains the
           change is your move (or type it: e5, Nf6, e7e5).
  the arm  Stockfish picks a move. The best move whose squares the arm can reach
           is played: captured piece to the drop zone first, castling as two moves.
           If no legal move is reachable you make the arm's move by hand.
           Afterwards the arm parks and the board is read again to verify.

The position is tracked in software from the start position; the camera only
sees occupancy. A reading that disagrees with the tracked position stops the
game until the board is fixed.

Setup (each session):  aim the camera, calibrate_board.py, calibrate_nadir.py,
then `board_state.py --ref` on the EMPTY board, then set up the pieces and park
the arm.

Usage:  .venv/bin/python chess/play.py [--skill 5] [--think 1.0] [--no-arm] [--policy P --ensemble]
        --no-arm: no robot; you make the arm's moves by hand (camera still used)
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import chess
import chess.engine
import cv2
import numpy as np

from board import FILES, draw_patches, load_calib, square_patch, warp_board
from board_state import REF_PATH, classify
from hover_squares import DROP

ARM = chess.WHITE
ENGINE = "stockfish"
READ_FRAMES = 5  # majority vote over this many frames...
READ_SPACING_S = 0.2  # ...this far apart, after a settle so a hand leaving the view is not read
READ_SETTLE_S = 0.5
LOG_DIR = Path(__file__).parent / "run_logs"


# ---- pure logic (no hardware) -------------------------------------------------------------

def occupancy(board):
    """{'e4': 'E'|'W'|'B'} for the tracked position."""
    out = {}
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        out[chess.square_name(sq)] = "E" if p is None else ("W" if p.color == chess.WHITE else "B")
    return out


def labels_to_occupancy(labels):
    """board_state.classify() labels [file][rank] -> {'e4': 'W', ...}."""
    return {f"{FILES[f]}{r + 1}": labels[f][r] for f in range(8) for r in range(8)}


def differences(expected, observed):
    """[(square, expected, observed)] where the two occupancies disagree, in board order."""
    return [(sq, expected[sq], observed[sq]) for sq in sorted(expected, key=lambda s: (s[1], s[0])) if expected[sq] != observed[sq]]


def matching_moves(board, observed):
    """Legal moves whose result has exactly the observed occupancy."""
    out = []
    for mv in board.legal_moves:
        board.push(mv)
        same = occupancy(board) == observed
        board.pop()
        if same:
            out.append(mv)
    return out


def captured_square(board, move):
    if board.is_en_passant(move):
        return chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
    return move.to_square


def arm_steps(board, move, reachable):
    """The (src, dst) pick-and-place steps the arm needs for a move, or None if it cannot do it.

    Captures go to the DROP zone first; castling is the king then the rook; promotions
    need a piece swap the arm cannot do."""
    if move.promotion:
        return None
    steps = []
    if board.is_capture(move):
        steps.append((chess.square_name(captured_square(board, move)), DROP))
    steps.append((chess.square_name(move.from_square), chess.square_name(move.to_square)))
    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        rook_from, rook_to = ("h", "f") if chess.square_file(move.to_square) == 6 else ("a", "d")
        steps.append((f"{rook_from}{rank + 1}", f"{rook_to}{rank + 1}"))
    squares = {sq for st in steps for sq in st if sq != DROP}
    return steps if squares <= set(reachable) else None


def choose_move(engine, board, reachable, think):
    """(move, steps): the engine's move if the arm can play it, else the best reachable
    alternative; steps is None when nothing reachable is legal (the human plays the move)."""
    # the e2 pawn is the arm's least reliable grasp: do not open the game with it
    moves = [m for m in board.legal_moves if not (board.fullmove_number == 1 and m.from_square == chess.E2)]
    best = engine.play(board, chess.engine.Limit(time=think), root_moves=moves).move
    steps = arm_steps(board, best, reachable)
    if steps is not None:
        return best, steps
    infos = engine.analyse(board, chess.engine.Limit(time=think), multipv=board.legal_moves.count())
    for info in infos:
        mv = info["pv"][0]
        steps = arm_steps(board, mv, reachable)
        if steps is not None:
            return mv, steps
    return best, None


def parse_move(board, text):
    for parse in (board.parse_san, board.parse_uci):
        try:
            return parse(text)
        except ValueError:
            pass
    return None


# ---- camera -------------------------------------------------------------------------------

class BoardReader:
    """Occupancy from a lerobot OpenCVCamera (RGB frames) using the board_state classifier."""

    def __init__(self, cam):
        calib = load_calib()
        if "nadir" not in calib:
            sys.exit("no nadir in board_calib.json; run calibrate_nadir.py first")
        if not REF_PATH.exists():
            sys.exit("no board_ref.png; run board_state.py --ref on the empty board first")
        self.cam, self.corners, self.nadir = cam, calib["corners"], calib["nadir"]
        self.ref = cv2.imread(str(REF_PATH))

    def read(self, n=READ_FRAMES):
        votes = {f"{FILES[f]}{r + 1}": Counter() for f in range(8) for r in range(8)}
        time.sleep(READ_SETTLE_S)
        for i in range(n):
            if i:
                time.sleep(READ_SPACING_S)
            frame = cv2.cvtColor(self.cam.read_latest(max_age_ms=1000), cv2.COLOR_RGB2BGR)
            self.last = warp_board(frame, self.corners)
            labels, _ = classify(self.last, self.ref, self.nadir)
            for sq, label in labels_to_occupancy(labels).items():
                votes[sq][label] += 1
        return {sq: c.most_common(1)[0][0] for sq, c in votes.items()}


def log_misread(reader, squares):
    """Save the frame behind a reading that disagrees (sample patches drawn) and print, per square, how far
    its patch is from the empty-board reference: the evidence for why an empty square read as a piece."""
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"misread_{time.strftime('%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(path), draw_patches(reader.last, reader.nadir))
    lab = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    now, ref = lab(reader.last), lab(reader.ref)
    for sq in squares:
        f, r = FILES.index(sq[0]), int(sq[1]) - 1
        d = square_patch(now, f, r, reader.nadir).reshape(-1, 3).mean(0) - square_patch(ref, f, r, reader.nadir).reshape(-1, 3).mean(0)
        print(f"   {sq}: brightness {d[0]:+.0f}, colour {np.linalg.norm(d[1:]):.1f} vs the empty reference (a piece is beyond 25 / 5)")
    print(f"   frame saved: {path.name}")


def print_board(board):
    print("\n" + str(board).replace(".", "·") + "\n")


def wait_until_matches(board, reader, what):
    """Re-read until the camera agrees with the tracked position; the human fixes the board in between."""
    while True:
        d = differences(occupancy(board), reader.read())
        if not d:
            return
        print(f"{what}: the camera disagrees with the tracked position on " + ", ".join(f"{sq} (expected {e}, saw {o})" for sq, e, o in d))
        log_misread(reader, [sq for sq, _, _ in d])
        input("Fix the board (or the lighting) and press Enter to read again... ")


# ---- turns --------------------------------------------------------------------------------

def human_turn(board, reader):
    while True:
        text = input("Your move: make it on the board and press Enter, or type it (e5, Nf6, e7e5): ").strip()
        if text:
            mv = parse_move(board, text)
            if mv is None:
                print("   not a legal move here")
                continue
            return mv
        observed = reader.read()
        cands = matching_moves(board, observed)
        if len(cands) == 1:
            return cands[0]
        if cands and len({(m.from_square, m.to_square) for m in cands}) == 1:  # promotion: the camera cannot tell the piece
            text = input(f"   promotion on {chess.square_name(cands[0].to_square)}: which piece? [q/r/b/n] ").strip().lower() or "q"
            return next((m for m in cands if chess.piece_symbol(m.promotion) == text), cands[0])
        d = differences(occupancy(board), observed)
        if not d:
            print("   no change seen on the board")
        else:
            print("   could not read your move; squares that changed: " + ", ".join(f"{sq} {e}->{o}" for sq, e, o in d))
            log_misread(reader, [sq for sq, _, _ in d])
        print("   make sure your hand is out of the view, or type the move")


def arm_turn(board, reader, engine, mover, reachable, think):
    mv, steps = choose_move(engine, board, reachable, think)
    san = board.san(mv)
    if steps is None or mover is None:
        why = "" if mover else " (no arm)"
        if steps is None and mover is not None:
            why = " (out of the arm's reach)"
        input(f"Arm plays {san}{why}: please make this move for the arm, then press Enter... ")
        return mv
    print(f"Arm plays {san}: " + ", ".join(f"{s}->{d}" for s, d in steps))
    for src, dst in steps:
        if not mover.move(src, dst):
            input(f"The arm could not move {src}->{dst}. Hold the piece if the jaws have it and press Enter (the jaws open)... ")
            mover.open_jaws()
            input("Finish the arm's move by hand, then press Enter... ")
            break
    mover.park()
    return mv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", type=int, default=5, help="Stockfish Skill Level 0..20")
    ap.add_argument("--think", type=float, default=1.0, help="seconds per engine move")
    ap.add_argument("--no-arm", action="store_true", help="no robot: you make the arm's moves by hand")
    ap.add_argument("--policy", default="policies/act_pick_v1")
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--max-s", type=float, default=15.0, help="policy time limit per pick attempt")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--ensemble", action="store_true", help="re-plan every frame (needed by the wrist-only policies)")
    a = ap.parse_args()

    mover = None
    if a.no_arm:
        from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

        from cameras import OVERHEAD, camera_index

        cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=camera_index(OVERHEAD), fps=30, width=640, height=480))
        cam.connect()
        reachable = set()
        close = cam.disconnect
    else:
        from run_pick import Mover

        mover = Mover(a.policy, a.ckpt, max_s=a.max_s, retries=a.retries, ensemble=a.ensemble)
        cam = mover.top.cam
        reachable = mover.squares
        close = mover.close
    reader = BoardReader(cam)

    engine = chess.engine.SimpleEngine.popen_uci(ENGINE)
    engine.configure({"Skill Level": a.skill})
    board = chess.Board()
    try:
        print("Set up the start position, arm parked, hands out of the view.")
        wait_until_matches(board, reader, "start position")
        while not board.is_game_over():
            print_board(board)
            if board.turn == ARM:
                mv = arm_turn(board, reader, engine, mover, reachable, a.think)
                board.push(mv)
                wait_until_matches(board, reader, "after the arm's move")
            else:
                mv = human_turn(board, reader)
                print(f"You played {board.san(mv)}")
                board.push(mv)
        print_board(board)
        print(f"Game over: {board.result()}  ({board.outcome().termination.name.lower()})")
    finally:
        engine.quit()
        close()


if __name__ == "__main__":
    main()
