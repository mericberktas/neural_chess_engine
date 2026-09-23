"""Board/move <-> tensor encoding shared by data prep, training, and inference.

21-channel scheme: the Maia-2 paper's 18-channel base (12 piece planes,
1 side-to-move plane, 4 castling-rights planes, 1 en-passant plane) plus 3
short-term-context planes added for run5 -- a mobility mask (channel 18:
squares holding a piece with >=1 legal move) and the previous move's
from/to squares (channels 19-20, all-zero on the game's first move). See
docs/reference/Egitim_Kosulari_Karsilastirma.md for why.
"""
import chess
import numpy as np

NUM_CHANNELS = 21

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
    for move in board.legal_moves:
        rank, file = divmod(move.from_square, 8)
        tensor[18, rank, file] = 1
    if board.move_stack:
        last_move = board.peek()
        rank, file = divmod(last_move.from_square, 8)
        tensor[19, rank, file] = 1
        rank, file = divmod(last_move.to_square, 8)
        tensor[20, rank, file] = 1
    return tensor


def move_to_indices(move: chess.Move) -> tuple[int, int]:
    """(from_square, to_square), each 0-63 — matches the two 64-way policy heads."""
    return move.from_square, move.to_square
