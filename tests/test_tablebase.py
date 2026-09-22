"""Assert-based self-check for src/tablebase.py. No framework: python tests/test_tablebase.py

Uses the handful of 3/4-piece Syzygy files checked into data/tablebase/
(KRvK, KBNvK, KPvK, KPvKP — a few hundred KB total, downloaded from
tablebase.lichess.ovh) that cover exactly the endgames below.
"""
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tablebase import DEFAULT_PIECE_THRESHOLD, best_move, open_tablebase, should_probe

TABLEBASE_DIR = Path(__file__).resolve().parent.parent / "data" / "tablebase"


def _assert_dtz_sound(fen: str):
    """best_move must reach the best WDL outcome any legal move could reach
    (i.e. never throw away a win or turn a draw into a loss)."""
    tb = open_tablebase(str(TABLEBASE_DIR))
    board = chess.Board(fen)

    best_reachable = max(-tb.probe_wdl(_push_pop(board, mv)) for mv in board.legal_moves)

    move = best_move(board, tb)
    assert move is not None
    assert move in board.legal_moves

    board.push(move)
    achieved = -tb.probe_wdl(board)
    tb.close()
    assert achieved == best_reachable, f"{fen}: got wdl {achieved}, best reachable was {best_reachable}"


def _push_pop(board, move):
    board.push(move)
    try:
        return board.copy(stack=False)
    finally:
        board.pop()


def test_kbn_vs_k_mate():
    # KBNvK — the textbook "hardest" mate. White to move, clearly winning.
    _assert_dtz_sound("8/8/8/8/3k3B/8/8/2K4N w - - 0 1")


def test_kr_vs_k_mate():
    # KRvK — basic rook mate. White to move, clearly winning.
    _assert_dtz_sound("4k3/8/4K3/8/8/8/8/7R w - - 0 1")


def test_pawn_race():
    # KPvKP pawn race: White's c-pawn is faster (wdl 2), but most king moves
    # throw the win away (some even lose outright) — only c4c5 keeps it.
    # This is the discriminating case: verifies best_move doesn't just play
    # *a* legal move but the one that preserves the winning outcome.
    fen = "8/1K4p1/8/8/2P5/k7/8/8 w - - 0 1"
    board = chess.Board(fen)
    tb = open_tablebase(str(TABLEBASE_DIR))
    assert tb.probe_wdl(board) == 2  # white is winning before any move

    move = best_move(board, tb)
    board.push(move)
    assert tb.probe_wdl(board) == -2  # still winning (opponent unconditionally lost)
    tb.close()


def test_should_probe_threshold():
    assert should_probe(chess.Board("4k3/8/4K3/8/8/8/8/7R w - - 0 1"))  # 3 pieces
    assert not should_probe(chess.Board())  # 32 pieces, starting position
    assert not should_probe(chess.Board(), piece_threshold=2)


def test_missing_table_returns_none_not_crash():
    # KQvK — no local table file for this material, only KRvK/KBNvK/KPvK/KPvKP.
    board = chess.Board("4k3/8/4K3/8/8/8/8/3Q4 w - - 0 1")
    tb = open_tablebase(str(TABLEBASE_DIR))
    assert best_move(board, tb) is None
    tb.close()


if __name__ == "__main__":
    test_kbn_vs_k_mate()
    test_kr_vs_k_mate()
    test_pawn_race()
    test_should_probe_threshold()
    test_missing_table_returns_none_not_crash()
    print("OK - all tablebase checks passed")
