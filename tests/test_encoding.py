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


if __name__ == "__main__":
    test_startpos_planes()
    test_en_passant_plane()
    test_move_to_indices()
    print("OK - all encoding checks passed")
