"""Assert-based self-check for src/model.py. No framework: python tests/test_model.py"""
import sys
import tempfile
from pathlib import Path

import chess
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from encoding import NUM_CHANNELS, board_to_tensor
from model import ChessTransformer, GeometricAttentionBias, legal_move_mask, select_legal_move

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


def test_gab_output_shape():
    gab = GeometricAttentionBias(in_channels=NUM_CHANNELS, nhead=2, d_compress=8, d_templates=4)
    boards = torch.randn(4, NUM_CHANNELS, 8, 8)
    bias = gab(boards)
    assert bias.shape == (4, 2, 64, 64)


def test_gab_bias_is_dynamic():
    # Same GAB instance, two different board states -> different bias.
    # Guards against a GAB that ends up ignoring its input (e.g. a bug that
    # only ever reads a constant slice) and produces a static bias.
    gab = GeometricAttentionBias(in_channels=NUM_CHANNELS, nhead=2, d_compress=8, d_templates=4)
    gab.eval()
    start = torch.from_numpy(board_to_tensor(chess.Board())).float().unsqueeze(0)
    board = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3"):
        board.push(chess.Move.from_uci(uci))
    later = torch.from_numpy(board_to_tensor(board)).float().unsqueeze(0)
    with torch.no_grad():
        bias_start = gab(start)
        bias_later = gab(later)
    assert not torch.allclose(bias_start, bias_later)


def test_encoder_mask_survives_eval_and_no_grad():
    # Regression test for a real bug found while building GAB (torch
    # 2.14.0+cpu): nn.TransformerEncoder's fused eval+no_grad "fast path"
    # silently produced NaN given a float attn_mask, while the same call
    # under grad-tracking (or in train mode) was correct. model.py disables
    # the fast path globally (torch.backends.mha.set_fastpath_enabled(False))
    # on import; this pins that behavior down so a future torch upgrade that
    # reintroduces the bug (or removes the workaround's effect) gets caught.
    d_model, nhead, num_layers, L, batch = 16, 2, 2, 72, 3
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=32, dropout=0.0, batch_first=True,
    )
    encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    torch.manual_seed(0)
    src = torch.randn(batch, L, d_model)
    mask = torch.randn(batch * nhead, L, L)

    with torch.no_grad():
        out_train_mode = encoder(src, mask=mask)
    encoder.eval()
    with torch.no_grad():
        out_eval_no_grad = encoder(src, mask=mask)

    assert not torch.isnan(out_eval_no_grad).any()
    assert torch.allclose(out_train_mode, out_eval_no_grad, atol=1e-5)

    with torch.no_grad():
        out_no_mask = encoder(src, mask=None)
    assert not torch.allclose(out_eval_no_grad, out_no_mask)  # mask must actually change the output


if __name__ == "__main__":
    test_forward_shapes()
    test_legal_move_mask_startpos()
    test_select_legal_move_is_always_legal()
    test_checkpoint_roundtrip()
    test_gab_output_shape()
    test_gab_bias_is_dynamic()
    test_encoder_mask_survives_eval_and_no_grad()
    print("OK - all model checks passed")
