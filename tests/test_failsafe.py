"""Assert-based self-check for src/failsafe.py. No framework: python tests/test_failsafe.py"""
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from failsafe import hangs_material, pick_safe_move


def test_hanging_queen_flagged_and_filter_falls_back():
    # White Qd1, Black king e8 + pawn e6. Qd1-d5 walks into a square only the
    # e6 pawn attacks and nothing defends -- a free queen for Black. Qd1-d2
    # is a quiet, unattacked square -- perfectly safe.
    board = chess.Board("4k3/8/4p3/8/8/8/8/3QK3 w - - 0 1")
    hangs_move = chess.Move.from_uci("d1d5")
    safe_move = chess.Move.from_uci("d1d2")
    assert hangs_move in board.legal_moves and safe_move in board.legal_moves

    assert hangs_material(board, hangs_move) is True
    assert hangs_material(board, safe_move) is False

    move, rejected = pick_safe_move(board, [hangs_move, safe_move])
    assert move == safe_move
    assert rejected == 1  # one higher-ranked candidate was rejected before this one


def test_ordinary_moves_and_equal_trades_not_flagged():
    # Quiet developing move, nothing attacks e4.
    start = chess.Board()
    assert hangs_material(start, chess.Move.from_uci("e2e4")) is False

    # White Nc3 x Nd5, recaptured by the e6 pawn -- an even knight-for-knight
    # trade. A check that flags every capture/recapture would wrongly flag
    # this; it must not.
    board = chess.Board("4k3/8/4p3/3n4/8/2N5/8/4K3 w - - 0 1")
    trade_move = chess.Move.from_uci("c3d5")
    assert trade_move in board.legal_moves
    assert hangs_material(board, trade_move) is False


def test_legitimate_sacrifice_not_overly_conservative():
    # White Qd4 x Qd5: Black's queen is defended by the c6 pawn, but this is
    # a straight queen-for-queen trade (captured value == piece value), not
    # a free loss -- must not be flagged even though the destination square
    # is still attacked afterwards.
    board = chess.Board("4k3/8/2p5/3q4/3Q4/8/8/4K3 w - - 0 1")
    capture_move = chess.Move.from_uci("d4d5")
    assert capture_move in board.legal_moves
    assert hangs_material(board, capture_move) is False


def test_no_safe_candidate_falls_back_to_top_with_flag():
    # Every candidate hangs material -- refusing to move isn't an option, so
    # pick_safe_move must still return a move (the top one), flagged via -1.
    board = chess.Board("4k3/8/4p3/8/8/8/8/3QK3 w - - 0 1")
    hangs_move = chess.Move.from_uci("d1d5")
    move, rejected = pick_safe_move(board, [hangs_move])
    assert move == hangs_move
    assert rejected == -1


if __name__ == "__main__":
    test_hanging_queen_flagged_and_filter_falls_back()
    test_ordinary_moves_and_equal_trades_not_flagged()
    test_legitimate_sacrifice_not_overly_conservative()
    test_no_safe_candidate_falls_back_to_top_with_flag()
    print("OK - all failsafe checks passed")
