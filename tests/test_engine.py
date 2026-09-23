"""Assert-based self-check for src/engine.py. No framework: python tests/test_engine.py

Uses the same small Syzygy files as test_tablebase.py (data/tablebase/).
"""
import sys
import tempfile
from pathlib import Path

import chess
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from engine import NeuralChessEngine, _deprioritize_drawing_moves
from model import ChessTransformer
from tablebase import best_move as tablebase_best_move
from tablebase import open_tablebase

TABLEBASE_DIR = Path(__file__).resolve().parent.parent / "data" / "tablebase"
TINY_MODEL_ARGS = {"d_model": 8, "nhead": 2, "num_layers": 1, "dim_feedforward": 16, "dropout": 0.0}


def _make_tiny_checkpoint(path: Path) -> None:
    model = ChessTransformer(**TINY_MODEL_ARGS)
    torch.save({"model_state_dict": model.state_dict(), "model_args": TINY_MODEL_ARGS}, path)


def test_tablebase_path_takes_priority_and_matches_direct_probe():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "tiny.pt"
        _make_tiny_checkpoint(ckpt_path)
        engine = NeuralChessEngine(str(ckpt_path), tablebase_dir=str(TABLEBASE_DIR))

        board = chess.Board("4k3/8/4K3/8/8/8/8/7R w - - 0 1")  # KRvK, 3 pieces
        expected_tb = open_tablebase(str(TABLEBASE_DIR))
        expected = tablebase_best_move(board, expected_tb)
        expected_tb.close()

        move = engine.select_move(board)
        assert move == expected  # engine must defer to the tablebase, not the (untrained) ANN


def test_ann_path_used_when_tablebase_unavailable():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "tiny.pt"
        _make_tiny_checkpoint(ckpt_path)
        engine = NeuralChessEngine(str(ckpt_path))  # no tablebase_dir at all

        board = chess.Board()  # 32 pieces, well above any tablebase threshold
        move = engine.select_move(board)
        assert move in board.legal_moves


def test_failsafe_wiring_falls_back_and_counts_the_trigger():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "tiny.pt"
        _make_tiny_checkpoint(ckpt_path)
        engine = NeuralChessEngine(str(ckpt_path))
        assert engine.failsafe_triggers == 0

        # Same scenario as tests/test_failsafe.py: Qd1-d5 hangs the queen for
        # free, Qd1-d2 is safe. Stub the ANN ranking so the wiring (not the
        # already-tested failsafe logic itself) is what's under test here.
        board = chess.Board("4k3/8/4p3/8/8/8/8/3QK3 w - - 0 1")
        hangs_move = chess.Move.from_uci("d1d5")
        safe_move = chess.Move.from_uci("d1d2")
        engine._policy_candidates = lambda b: [hangs_move, safe_move]

        move = engine.select_move(board)
        assert move == safe_move
        assert engine.failsafe_triggers == 1


def test_deprioritizes_a_candidate_that_triggers_threefold_repetition():
    board = chess.Board("k7/8/8/8/8/8/8/K7 w - - 0 1")
    # Shuffle kings back and forth; after these 7 plies, Black's Kb8-a8 would
    # recreate the starting position for the 3rd time.
    for uci in ["a1b1", "a8b8", "b1a1", "b8a8", "a1b1", "a8b8", "b1a1"]:
        board.push(chess.Move.from_uci(uci))

    repeating_move = chess.Move.from_uci("b8a8")
    other_move = chess.Move.from_uci("b8c8")

    board.push(repeating_move)
    assert board.is_repetition(3)
    board.pop()
    board.push(other_move)
    assert not board.is_repetition(3)
    board.pop()

    ordered = _deprioritize_drawing_moves(board, [repeating_move, other_move])
    assert ordered == [other_move, repeating_move]

    # If every candidate repeats, the order is left unchanged (no legal
    # alternative exists to prefer).
    ordered_all_repeating = _deprioritize_drawing_moves(board, [repeating_move])
    assert ordered_all_repeating == [repeating_move]


def test_deprioritizes_a_candidate_that_stalemates_the_opponent():
    # Real position from a live meric_bot game (meriicnumber2 vs meric_bot,
    # 2026-09-23): Black to move, hugely winning (Q+2P vs bare K), but 6 of
    # its 29 legal moves immediately stalemate White -- e5e4 (the one the
    # ANN actually played, live, drawing a dead-won game) is one of them.
    board = chess.Board("3K4/1p6/2q2kp1/4p3/8/8/8/8 b - - 7 51")
    stalemating_move = chess.Move.from_uci("e5e4")
    safe_move = chess.Move.from_uci("c6d7")  # Qd7# -- an actual mate, still safer to just check it's non-drawing

    assert stalemating_move in board.legal_moves
    board.push(stalemating_move)
    assert board.is_stalemate()
    board.pop()

    ordered = _deprioritize_drawing_moves(board, [stalemating_move, safe_move])
    assert ordered == [safe_move, stalemating_move]


if __name__ == "__main__":
    test_tablebase_path_takes_priority_and_matches_direct_probe()
    test_ann_path_used_when_tablebase_unavailable()
    test_failsafe_wiring_falls_back_and_counts_the_trigger()
    test_deprioritizes_a_candidate_that_triggers_threefold_repetition()
    test_deprioritizes_a_candidate_that_stalemates_the_opponent()
    print("OK - all engine checks passed")
