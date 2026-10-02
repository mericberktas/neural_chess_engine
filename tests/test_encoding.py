"""Assert-based self-check for src/encoding.py. No framework: python tests/test_encoding.py"""
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from encoding import NUM_CHANNELS, board_to_tensor, move_to_indices, position_extras


def test_startpos_planes():
    t = board_to_tensor(chess.Board())
    assert t.shape == (NUM_CHANNELS, 8, 8)
    assert (t[0, 1, :] == 1).all()   # white pawns, rank 2
    assert (t[6, 6, :] == 1).all()   # black pawns, rank 7
    assert t[5, 0, 4] == 1           # white king, e1
    assert t[11, 7, 4] == 1          # black king, e8
    assert (t[12] == 1).all()        # white to move
    assert (t[13:17] == 1).all()     # all four castling rights available
    assert (t[17] == 0).all()        # no en passant target


def test_en_passant_plane():
    board = chess.Board()
    for uci in ("e2e4", "a7a6", "e4e5", "d7d5"):
        board.push(chess.Move.from_uci(uci))
    t = board_to_tensor(board)
    rank, file = divmod(board.ep_square, 8)
    assert t[17, rank, file] == 1
    assert t[17].sum() == 1


def test_move_to_indices():
    assert move_to_indices(chess.Move.from_uci("e2e4")) == (12, 28)


def test_mobility_plane_startpos():
    t = board_to_tensor(chess.Board())
    assert (t[18, 1, :] == 1).all()  # all 8 pawns can move
    assert t[18, 0, 1] == 1 and t[18, 0, 6] == 1  # Nb1, Ng1 can move
    assert t[18, 0, [0, 2, 3, 4, 5, 7]].sum() == 0  # rooks/bishops/queen/king still blocked
    assert t[18, 2:].sum() == 0  # nothing else on the board yet


def test_last_move_planes_empty_at_start_and_set_after_a_move():
    start = board_to_tensor(chess.Board())
    assert (start[19] == 0).all()
    assert (start[20] == 0).all()

    board = chess.Board()
    for uci in ("e2e4", "a7a6", "e4e5", "d7d5"):
        board.push(chess.Move.from_uci(uci))
    t = board_to_tensor(board)
    from_rank, from_file = divmod(chess.D7, 8)
    to_rank, to_file = divmod(chess.D5, 8)
    assert t[19, from_rank, from_file] == 1 and t[19].sum() == 1
    assert t[20, to_rank, to_file] == 1 and t[20].sum() == 1


def test_see_risk_plane():
    # Nc2 attacks Black Qd4 (a knight move); Qd4 doesn't attack back (not a
    # queen-line), so it's cleanly undefended -> full value lost.
    undefended = chess.Board("4k3/8/8/8/3q4/8/2N5/4K3 w - - 0 1")
    t = board_to_tensor(undefended)
    rank, file = divmod(chess.D4, 8)
    assert t[21, rank, file] == 9
    rank, file = divmod(chess.C2, 8)
    assert t[21, rank, file] == 0  # attacker itself isn't attacked back

    # Nc3 attacks Black Qd5, defended by the e6 pawn -> queen_value - knight_value.
    defended = chess.Board("4k3/8/4p3/3q4/8/2N5/8/4K3 w - - 0 1")
    t = board_to_tensor(defended)
    rank, file = divmod(chess.D5, 8)
    assert t[21, rank, file] == 9 - 3

    # Nc3 vs Nd5, defended by the e6 pawn -- an even trade nets to 0, floored.
    even_trade = chess.Board("4k3/8/4p3/3n4/8/2N5/8/4K3 w - - 0 1")
    t = board_to_tensor(even_trade)
    rank, file = divmod(chess.D5, 8)
    assert t[21, rank, file] == 0

    # White king in check from a rook -- kings are never flagged, even under attack.
    king_in_check = chess.Board("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")
    t = board_to_tensor(king_in_check)
    rank, file = divmod(chess.E1, 8)
    assert t[21, rank, file] == 0


def test_position_extras_legal_mask_and_attacks():
    legal_packed, attacks = position_extras(chess.Board())
    legal = np.unpackbits(legal_packed, bitorder="little").astype(bool)
    assert legal.sum() == 20  # 16 pawn moves + 4 knight moves at the start
    assert legal[chess.E2 * 64 + chess.E4] and legal[chess.G1 * 64 + chess.F3]
    assert not legal[chess.E2 * 64 + chess.E5]

    bits = np.unpackbits(attacks, axis=1, bitorder="little")  # (64, 64): [from, to]
    assert bits[chess.G1].nonzero()[0].tolist() == sorted([chess.E2, chess.F3, chess.H3])  # incl. defended own pawn e2
    assert bits[chess.E4].sum() == 0  # empty square attacks nothing
    assert bits[chess.A1].nonzero()[0].tolist() == sorted([chess.A2, chess.B1])  # blocked rook: defends, no more


if __name__ == "__main__":
    test_startpos_planes()
    test_en_passant_plane()
    test_move_to_indices()
    test_mobility_plane_startpos()
    test_last_move_planes_empty_at_start_and_set_after_a_move()
    test_see_risk_plane()
    test_position_extras_legal_mask_and_attacks()
    print("OK - all encoding checks passed")
