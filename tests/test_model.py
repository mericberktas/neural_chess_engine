"""Assert-based self-check for src/model.py. No framework: python tests/test_model.py"""
import sys
import tempfile
from pathlib import Path

import chess
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from encoding import board_to_tensor
from model import ChessTransformer, legal_move_mask, select_legal_move

TINY_KWARGS = dict(d_model=32, nhead=2, num_layers=2, dim_feedforward=64, dropout=0.0)


def test_forward_shapes():
    model = ChessTransformer(**TINY_KWARGS)
    boards = torch.stack([torch.from_numpy(board_to_tensor(chess.Board())) for _ in range(4)])
    from_logits, to_logits = model(boards)
    assert from_logits.shape == (4, 64)
    assert to_logits.shape == (4, 64)
    assert torch.isfinite(from_logits).all() and torch.isfinite(to_logits).all()


def test_legal_move_mask_startpos():
    board = chess.Board()
    mask = legal_move_mask(board)
    assert mask.shape == (64, 64)
    assert mask.sum() == len(list(board.legal_moves)) == 20  # startpos: 20 legal moves, no from/to pair repeats

    # a known legal move: e2-e4
    e2, e4 = chess.E2, chess.E4
    assert mask[e2, e4]
    # a known illegal move: e2-e5 (pawn can't jump 3 ranks)
    e5 = chess.E5
    assert not mask[e2, e5]
    # no white piece starts a move from a black home-row square at move 1
    assert not mask[chess.E7, chess.E5]


def test_select_legal_move_is_always_legal():
    model = ChessTransformer(**TINY_KWARGS)
    model.eval()
    board = chess.Board()
    boards = torch.from_numpy(board_to_tensor(board)).unsqueeze(0)
    with torch.no_grad():
        from_logits, to_logits = model(boards)
    move = select_legal_move(from_logits[0], to_logits[0], board)
    assert move in board.legal_moves


def test_checkpoint_roundtrip():
    model = ChessTransformer(**TINY_KWARGS)
    boards = torch.stack([torch.from_numpy(board_to_tensor(chess.Board()))])
    with torch.no_grad():
        expected_from, expected_to = model(boards)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        torch.save({"model_state_dict": model.state_dict()}, path)

        loaded = ChessTransformer(**TINY_KWARGS)
        checkpoint = torch.load(path, weights_only=True)
        loaded.load_state_dict(checkpoint["model_state_dict"])

    for p1, p2 in zip(model.parameters(), loaded.parameters()):
        assert torch.equal(p1, p2)

    loaded.eval()
    with torch.no_grad():
        got_from, got_to = loaded(boards)
    # allclose, not equal: CPU matmul thread-reduction order can perturb the
    # last float bits between two separate forward passes even with identical weights.
    assert torch.allclose(expected_from, got_from, atol=1e-5)
    assert torch.allclose(expected_to, got_to, atol=1e-5)


if __name__ == "__main__":
    test_forward_shapes()
    test_legal_move_mask_startpos()
    test_select_legal_move_is_always_legal()
    test_checkpoint_roundtrip()
    print("OK - all model checks passed")
