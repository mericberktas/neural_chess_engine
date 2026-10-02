"""Board/move <-> tensor encoding shared by data prep, training, and inference.

22-channel scheme: the Maia-2 paper's 18-channel base (12 piece planes,
1 side-to-move plane, 4 castling-rights planes, 1 en-passant plane), 3
short-term-context planes added for run5 -- a mobility mask (channel 18:
squares holding a piece with >=1 legal move) and the previous move's
from/to squares (channels 19-20, all-zero on the game's first move) -- and
one static-exchange-evaluation-risk plane added for run6 (channel 21: for
each occupied square, the material a hanging/under-defended piece there
would lose to the opponent's cheapest attacker, 0-9, see
failsafe.hanging_loss_at). See docs/reference/Egitim_Kosulari_Karsilastirma.md
for why.
"""
import chess
import numpy as np

from failsafe import hanging_loss_at

NUM_CHANNELS = 22
SEE_CHANNEL = 21  # index of the static-exchange-risk plane (run6)

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
    for square in board.piece_map():
        rank, file = divmod(square, 8)
        tensor[SEE_CHANNEL, rank, file] = hanging_loss_at(board, square)
    return tensor


def position_extras(board: chess.Board) -> tuple[np.ndarray, np.ndarray]:
    """Per-position side data stored next to board_to_tensor's output (shard
    format v2), kept as packed bits so a position costs ~1KB instead of the
    ~20KB of the equivalent dense arrays.

    legal:   (512,) uint8 -- bit (from*64 + to) set iff some legal move uses
             that from/to pair (promotion piece not distinguished, same as
             model.legal_move_mask); little-endian bit order within each byte.
    attacks: (64, 8) uint8 -- row s is board.attacks_mask(s) as 8 little-endian
             bytes: bit (8*byte + k) set iff the piece on s attacks/defends
             that square (blockers respected; all zero for an empty s).
    """
    legal = np.zeros(64 * 64, dtype=bool)
    for move in board.legal_moves:
        legal[move.from_square * 64 + move.to_square] = True
    attacks = np.array([board.attacks_mask(sq) for sq in chess.SQUARES], dtype="<u8").view(np.uint8).reshape(64, 8)
    return np.packbits(legal, bitorder="little"), attacks


def move_to_indices(move: chess.Move) -> tuple[int, int]:
    """(from_square, to_square), each 0-63 — matches the two 64-way policy heads."""
    return move.from_square, move.to_square
