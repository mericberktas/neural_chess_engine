"""Board/move <-> tensor encoding shared by data prep, training, and inference.

18-channel scheme from the Maia-2 paper: 12 piece planes (type x color),
1 side-to-move plane, 4 castling-rights planes, 1 en-passant plane.
"""
import chess
import numpy as np

NUM_CHANNELS = 18

_PIECE_ORDER = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)


def board_to_tensor(board: chess.Board) -> np.ndarray:
    tensor = np.zeros((NUM_CHANNELS, 8, 8), dtype=np.uint8)
    for color in (chess.WHITE, chess.BLACK):
        offset = 0 if color == chess.WHITE else 6
        for i, piece_type in enumerate(_PIECE_ORDER):
            for square in board.pieces(piece_type, color):
                rank, file = divmod(square, 8)
                tensor[offset + i, rank, file] = 1
    tensor[12, :, :] = board.turn == chess.WHITE
    tensor[13, :, :] = board.has_kingside_castling_rights(chess.WHITE)
    tensor[14, :, :] = board.has_queenside_castling_rights(chess.WHITE)
    tensor[15, :, :] = board.has_kingside_castling_rights(chess.BLACK)
    tensor[16, :, :] = board.has_queenside_castling_rights(chess.BLACK)
    if board.ep_square is not None:
        rank, file = divmod(board.ep_square, 8)
        tensor[17, rank, file] = 1
    return tensor


def move_to_indices(move: chess.Move) -> tuple[int, int]:
    """(from_square, to_square), each 0-63 — matches the two 64-way policy heads."""
    return move.from_square, move.to_square
