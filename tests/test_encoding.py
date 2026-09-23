"""Assert-based self-check for src/encoding.py. No framework: python tests/test_encoding.py"""
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from encoding import NUM_CHANNELS, board_to_tensor, move_to_indices


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


if __name__ == "__main__":
    test_startpos_planes()
    test_en_passant_plane()
    test_move_to_indices()
    test_mobility_plane_startpos()
    test_last_move_planes_empty_at_start_and_set_after_a_move()
    print("OK - all encoding checks passed")
