"""Assert-based self-check for src/model.py. No framework: python tests/test_model.py"""
import sys
import tempfile
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from encoding import NUM_CHANNELS, board_to_tensor
from model import (
    ChessTransformer, GeometricAttentionBias, legal_move_mask, select_legal_move, top_k_legal_moves_from_joint,
    unpack_bits,
)

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


def test_per_layer_gab_matches_shared_gab_when_layers_are_identical():
    # Per-layer GAB with every layer's head_proj slice set equal to the shared
    # model's single head_proj must reproduce the shared model exactly --
    # pins down that the (B, num_layers, nhead, 64, 64) layout/indexing feeds
    # each encoder layer the right slice, and that the manual layer loop
    # matches the shared-mask path.
    boards = torch.stack([torch.from_numpy(board_to_tensor(chess.Board())) for _ in range(3)])
    torch.manual_seed(0)
    shared = ChessTransformer(**TINY_KWARGS).eval()
    per_layer = ChessTransformer(**TINY_KWARGS, gab_per_layer=True).eval()
    state = {k: v for k, v in shared.state_dict().items() if not k.startswith("gab.head_proj")}
    per_layer.load_state_dict(state, strict=False)
    layers = TINY_KWARGS["num_layers"]
    with torch.no_grad():
        per_layer.gab.head_proj.weight.copy_(shared.gab.head_proj.weight.repeat(layers, 1))
        per_layer.gab.head_proj.bias.copy_(shared.gab.head_proj.bias.repeat(layers))
        expected = shared(boards)
        got = per_layer(boards)
    assert torch.allclose(expected[0], got[0], atol=1e-5)
    assert torch.allclose(expected[1], got[1], atol=1e-5)


def test_per_layer_gab_layers_actually_differ():
    # Fresh (independently initialised) per-layer biases must give each layer
    # a different mask, not the same one repeated.
    gab = GeometricAttentionBias(in_channels=NUM_CHANNELS, nhead=2, d_compress=8, d_templates=4, num_outputs=3)
    boards = torch.randn(2, NUM_CHANNELS, 8, 8)
    bias = gab(boards)
    assert bias.shape == (2, 3, 2, 64, 64)
    assert not torch.allclose(bias[:, 0], bias[:, 1])


def test_shared_gab_checkpoint_layout_unchanged():
    # Default (gab_per_layer=False) must keep run6's exact parameter shapes,
    # or every pre-run9 checkpoint stops loading.
    model = ChessTransformer(**TINY_KWARGS)
    assert model.gab.head_proj.out_features == TINY_KWARGS["nhead"] * 32
    assert ChessTransformer(**TINY_KWARGS, gab_per_layer=True).gab.head_proj.out_features == (
        TINY_KWARGS["num_layers"] * TINY_KWARGS["nhead"] * 32
    )


def test_unpack_bits_inverts_numpy_packbits_little_endian():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(5, 64), dtype=np.uint8).astype(bool)
    packed = np.packbits(bits, axis=1, bitorder="little")
    got = unpack_bits(torch.from_numpy(packed))
    assert got.shape == (5, 64) and got.dtype == torch.bool
    assert (got.numpy() == bits).all()


def test_joint_logits_is_from_plus_to():
    model = ChessTransformer(**TINY_KWARGS).eval()
    boards = torch.stack([torch.from_numpy(board_to_tensor(chess.Board()))])
    with torch.no_grad():
        from_logits, to_logits = model(boards)
        joint = model.joint_logits(boards)
    assert joint.shape == (1, 64, 64)
    assert torch.allclose(joint[0, 12, 28], from_logits[0, 12] + to_logits[0, 28])


def test_pair_head_adds_a_term_that_is_not_additive_and_stays_off_by_default():
    boards = torch.stack([torch.from_numpy(board_to_tensor(chess.Board()))])
    torch.manual_seed(0)
    plain = ChessTransformer(**TINY_KWARGS).eval()
    paired = ChessTransformer(**TINY_KWARGS, pair_head=True).eval()
    assert not any(k.startswith("pair_") for k in plain.state_dict())  # run6/run9a layouts unchanged
    paired.load_state_dict(plain.state_dict(), strict=False)  # same trunk/heads, plus fresh pair_q/pair_k
    with torch.no_grad():
        additive = plain.joint_logits(boards)
        joint = paired.joint_logits(boards)
    assert joint.shape == (1, 64, 64)
    assert not torch.allclose(joint, additive)  # the bilinear term really contributes
    # additive part is rank-1 structure (from_i + to_j); the pair term must break it
    second_diff = joint[0, 0, 0] - joint[0, 0, 1] - joint[0, 1, 0] + joint[0, 1, 1]
    assert abs(second_diff.item()) > 1e-6
    assert abs((additive[0, 0, 0] - additive[0, 0, 1] - additive[0, 1, 0] + additive[0, 1, 1]).item()) < 1e-5


def test_top_k_from_joint_returns_distinct_legal_moves_with_pair_head():
    board = chess.Board()
    model = ChessTransformer(**TINY_KWARGS, pair_head=True).eval()
    boards = torch.from_numpy(board_to_tensor(board)).unsqueeze(0)
    with torch.no_grad():
        joint = model.joint_logits(boards)[0]
    moves = top_k_legal_moves_from_joint(joint, board, 5)
    assert len(moves) == 5 and len(set(moves)) == 5
    assert all(m in board.legal_moves for m in moves)


if __name__ == "__main__":
    test_forward_shapes()
    test_legal_move_mask_startpos()
    test_select_legal_move_is_always_legal()
    test_checkpoint_roundtrip()
    test_gab_output_shape()
    test_gab_bias_is_dynamic()
    test_encoder_mask_survives_eval_and_no_grad()
    test_per_layer_gab_matches_shared_gab_when_layers_are_identical()
    test_per_layer_gab_layers_actually_differ()
    test_shared_gab_checkpoint_layout_unchanged()
    test_unpack_bits_inverts_numpy_packbits_little_endian()
    test_joint_logits_is_from_plus_to()
    test_pair_head_adds_a_term_that_is_not_additive_and_stays_off_by_default()
    test_top_k_from_joint_returns_distinct_legal_moves_with_pair_head()
    print("OK - all model checks passed")
