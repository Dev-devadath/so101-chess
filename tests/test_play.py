"""Offline tests for the game-loop logic in chess/play.py (no arm, no camera).

Run:  .venv/bin/python tests/test_play.py
"""

import json
import sys
import unittest
from pathlib import Path

import chess
import chess.engine

sys.path.insert(0, str(Path(__file__).parent.parent / "chess"))

from hover_squares import DROP  # noqa: E402
from play import ENGINE, arm_steps, choose_move, differences, matching_moves, occupancy  # noqa: E402

REACHABLE = {s for s in json.load(open(Path(__file__).parent.parent / "chess" / "square_points.json")) if not s.startswith("_")}


def after(board, san):
    b = board.copy()
    b.push_san(san)
    return occupancy(b)


class Occupancy(unittest.TestCase):
    def test_start_position(self):
        occ = occupancy(chess.Board())
        self.assertEqual(sum(v == "W" for v in occ.values()), 16)
        self.assertEqual(sum(v == "B" for v in occ.values()), 16)
        self.assertEqual(occ["e1"], "W")
        self.assertEqual(occ["e8"], "B")
        self.assertEqual(occ["e4"], "E")

    def test_differences_in_board_order(self):
        b = chess.Board()
        self.assertEqual(differences(occupancy(b), after(b, "e4")), [("e2", "W", "E"), ("e4", "E", "W")])


class MatchHumanMove(unittest.TestCase):
    def test_plain_move_is_unique(self):
        b = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
        self.assertEqual(matching_moves(b, after(b, "e5")), [chess.Move.from_uci("e7e5")])

    def test_capture(self):
        b = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2")
        self.assertEqual(matching_moves(b, after(b, "exd4")), [chess.Move.from_uci("e5d4")])

    def test_castling_four_squares(self):
        b = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 4")
        self.assertEqual(matching_moves(b, after(b, "O-O")), [chess.Move.from_uci("e8g8")])

    def test_en_passant(self):
        b = chess.Board("rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3")
        self.assertEqual(matching_moves(b, after(b, "dxe3")), [chess.Move.from_uci("d4e3")])

    def test_promotion_is_ambiguous_only_in_piece(self):
        b = chess.Board("8/8/8/8/8/8/1k4p1/7K b - - 0 1")
        cands = matching_moves(b, after(b, "g1=Q"))
        self.assertEqual({m.from_square for m in cands}, {chess.G2})
        self.assertEqual({m.promotion for m in cands}, {chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT})

    def test_no_change_matches_nothing(self):
        b = chess.Board()
        self.assertEqual(matching_moves(b, occupancy(b)), [])


class ArmSteps(unittest.TestCase):
    def test_plain(self):
        b = chess.Board()
        self.assertEqual(arm_steps(b, chess.Move.from_uci("e2e4"), REACHABLE), [("e2", "e4")])

    def test_capture_goes_to_drop_first(self):
        b = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 2")
        self.assertEqual(arm_steps(b, chess.Move.from_uci("d4e5"), REACHABLE), [("e5", DROP), ("d4", "e5")])

    def test_en_passant_captured_pawn_square(self):
        b = chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
        self.assertEqual(arm_steps(b, chess.Move.from_uci("e5f6"), REACHABLE | {"f6"}), [("f5", DROP), ("e5", "f6")])

    def test_castling_is_king_then_rook(self):
        b = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 1")
        self.assertEqual(arm_steps(b, chess.Move.from_uci("e1g1"), REACHABLE), [("e1", "g1"), ("h1", "f1")])
        b = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/R3KBNR w KQkq - 0 1")
        self.assertEqual(arm_steps(b, chess.Move.from_uci("e1c1"), REACHABLE), [("e1", "c1"), ("a1", "d1")])

    def test_out_of_reach(self):
        b = chess.Board("rnbqkbnr/pppppppp/8/8/8/4N3/PPPPPPPP/RNBQKB1R w KQkq - 0 1")
        self.assertEqual(arm_steps(b, chess.Move.from_uci("e3f5"), REACHABLE), [("e3", "f5")])  # rank 5 is taught
        self.assertIsNone(arm_steps(b, chess.Move.from_uci("e3d5"), REACHABLE - {"d5"}))
        b = chess.Board("rnbqkbnr/pppppppp/8/8/4N3/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1")
        self.assertIsNone(arm_steps(b, chess.Move.from_uci("e4f6"), REACHABLE))  # rank 6 is not
        b = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        self.assertIsNone(arm_steps(b, chess.Move.from_uci("b1c3"), REACHABLE - {"c3"}))

    def test_promotion_never(self):
        b = chess.Board("8/P6k/8/8/8/8/8/K7 w - - 0 1")
        self.assertIsNone(arm_steps(b, chess.Move.from_uci("a7a8q"), REACHABLE | {"a7", "a8"}))


class Engine(unittest.TestCase):
    def test_stockfish_prefers_reachable_moves(self):
        engine = chess.engine.SimpleEngine.popen_uci(ENGINE)
        try:
            engine.configure({"Skill Level": 5})
            b = chess.Board()
            mv, steps = choose_move(engine, b, REACHABLE, 0.2)
            self.assertIn(mv, b.legal_moves)
            self.assertIsNotNone(steps)
            # a position where the only sensible moves are far away: still returns a legal move
            b = chess.Board("k7/8/8/8/8/8/8/K6R w - - 0 1")
            mv, steps = choose_move(engine, b, {"h1", "h2"}, 0.2)
            self.assertIn(mv, b.legal_moves)
            self.assertIn(steps, ([("h1", "h2")], None))
        finally:
            engine.quit()


if __name__ == "__main__":
    unittest.main()
