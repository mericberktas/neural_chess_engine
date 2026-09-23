"""Assert-based self-check for src/engine.py. No framework: python tests/test_engine.py

Uses the same small Syzygy files as test_tablebase.py (data/tablebase/).
"""
import sys
import tempfile
from pathlib import Path

import chess
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from engine import NeuralChessEngine
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


if __name__ == "__main__":
    test_tablebase_path_takes_priority_and_matches_direct_probe()
    test_ann_path_used_when_tablebase_unavailable()
    test_failsafe_wiring_falls_back_and_counts_the_trigger()
    print("OK - all engine checks passed")
