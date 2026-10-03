"""Assert-based self-check for src/model.py. No framework: python tests/test_model.py"""
import sys
import tempfile
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from encoding import NUM_CHANNELS, SEE_CHANNEL, board_to_tensor, position_extras
from model import ChessTransformer, GeometricAttentionBias, legal_move_mask, select_legal_move, unpack_bits

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


def _startpos_inputs():
    board = chess.Board()
    boards = torch.from_numpy(board_to_tensor(board)).unsqueeze(0)
    attacks = torch.from_numpy(position_extras(board)[1]).unsqueeze(0)
    return boards, attacks


def test_attack_relation_classes_and_bias_term_land_on_the_right_pairs():
    boards, attacks = _startpos_inputs()
    piece_idx = ChessTransformer._piece_indices(boards.float())
    rel, mask = ChessTransformer._attack_relations(piece_idx, attacks)
    assert mask[0, chess.G1, chess.E2] and mask[0, chess.G1, chess.F3]
    assert not mask[0, chess.G1, chess.G2]
    knight_defends_pawn = (1 * 7 + 0) * 2 + 1  # attacker N, target P, same colour
    knight_covers_empty = (1 * 7 + 6) * 2 + 0  # attacker N, target empty
    assert rel[0, chess.G1, chess.E2] == knight_defends_pawn
    assert rel[0, chess.G1, chess.F3] == knight_covers_empty

    model = ChessTransformer(**TINY_KWARGS, attack_bias=True)
    with torch.no_grad():
        model.attack_embed.zero_()
        model.attack_embed[0, 0, knight_defends_pawn, :] = 1.0  # attacker -> target direction
        model.attack_embed[0, 1, knight_defends_pawn, :] = 2.0  # target <- attacker direction
    pairs = ChessTransformer._attack_pairs(rel, mask)
    term = model._attack_term(pairs, layer_idx=0)
    assert term.shape == (1, TINY_KWARGS["nhead"], 64, 64)
    assert (term[0, :, chess.G1, chess.E2] == 1.0).all()  # the knight attends to the pawn it defends
    assert (term[0, :, chess.E2, chess.G1] == 2.0).all()  # ...and the pawn attends back to its defender
    assert (term[0, :, chess.G1, chess.F3] == 0.0).all()  # empty-target relation weight left at 0
    assert (term[0, :, chess.A1, chess.H8] == 0.0).all()  # no attack, no bias
    assert (model._attack_term(pairs, layer_idx=1) == 0.0).all()  # other layers have their own weights


def test_attack_bias_starts_as_the_plain_model_then_reacts_to_attacks():
    boards, attacks = _startpos_inputs()
    torch.manual_seed(0)
    plain = ChessTransformer(**TINY_KWARGS, gab_per_layer=True).eval()
    attack = ChessTransformer(**TINY_KWARGS, gab_per_layer=True, attack_bias=True).eval()
    attack.load_state_dict(plain.state_dict(), strict=False)  # shared trunk, fresh zero attack_embed
    with torch.no_grad():
        assert torch.allclose(plain.joint_logits(boards), attack.joint_logits(boards, attacks), atol=1e-6)
        attack.attack_embed.normal_()
        assert not torch.allclose(plain.joint_logits(boards), attack.joint_logits(boards, attacks))
    try:
        attack.joint_logits(boards)
    except ValueError:
        return
    raise AssertionError("attack_bias model without attacks should raise")


def test_no_see_channel_equals_zeroing_that_plane():
    board = chess.Board("4k3/8/4p3/3Q4/8/8/8/4K3 w - - 0 1")  # undefended queen attacked by a pawn: SEE plane > 0
    boards = torch.from_numpy(board_to_tensor(board)).unsqueeze(0)
    zeroed = boards.clone()
    zeroed[:, SEE_CHANNEL] = 0
    assert boards[:, SEE_CHANNEL].any()
    torch.manual_seed(0)
    with_see = ChessTransformer(**TINY_KWARGS).eval()
    without_see = ChessTransformer(**TINY_KWARGS, see_channel=False).eval()
    without_see.load_state_dict(with_see.state_dict())  # channel_keep is a non-persistent buffer: layout unchanged
    with torch.no_grad():
        assert torch.allclose(without_see.joint_logits(boards), with_see.joint_logits(zeroed), atol=1e-6)
        assert not torch.allclose(with_see.joint_logits(boards), with_see.joint_logits(zeroed))


def test_sparse_attack_term_equals_the_dense_definition():
    # The scatter-based _attack_term must equal the straightforward dense
    # definition: weight(from->to) where from attacks to, plus weight(reverse)
    # at the transposed position, summed when both directions attack.
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    boards = torch.from_numpy(board_to_tensor(board)).unsqueeze(0)
    attacks = torch.from_numpy(position_extras(board)[1]).unsqueeze(0)
    model = ChessTransformer(**TINY_KWARGS, attack_bias=True)
    with torch.no_grad():
        model.attack_embed.normal_()
    rel, mask = ChessTransformer._attack_relations(ChessTransformer._piece_indices(boards.float()), attacks)
    for layer in range(TINY_KWARGS["num_layers"]):
        fwd = model.attack_embed[layer, 0][rel]
        bwd = model.attack_embed[layer, 1][rel.transpose(1, 2)]
        dense = (mask.unsqueeze(-1) * fwd + mask.transpose(1, 2).unsqueeze(-1) * bwd).permute(0, 3, 1, 2)
        assert torch.allclose(model._attack_term(ChessTransformer._attack_pairs(rel, mask), layer), dense, atol=1e-6)


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
    test_attack_relation_classes_and_bias_term_land_on_the_right_pairs()
    test_attack_bias_starts_as_the_plain_model_then_reacts_to_attacks()
    test_no_see_channel_equals_zeroing_that_plane()
    test_sparse_attack_term_equals_the_dense_definition()
    print("OK - all model checks passed")
