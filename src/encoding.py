"""Board/move <-> tensor encoding shared by data prep, training, and inference.

22-channel scheme: the Maia-2 paper's 18-channel base (12 piece planes,
1 side-to-move plane, 4 castling-rights planes, 1 en-passant plane), 3
short-term-context planes added for run5 -- a mobility mask (channel 18:
squares holding a piece with >=1 legal move) and the previous move's
from/to squares (channels 19-20, all-zero on the game's first move) -- and
one static-exchange-evaluation-risk plane added for run6 (channel 21: for
each occupied square, the material a hanging/under-defended piece there
would lose to the opponent's cheapest attacker, 0-9, see _see_risk). See
docs/reference/Egitim_Kosulari_Karsilastirma.md for why.
"""
import chess
import numpy as np

from failsafe import PIECE_VALUES

NUM_CHANNELS = 22

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
        tensor[21, rank, file] = _see_risk(board, square)
    return tensor


def _see_risk(board: chess.Board, square: int) -> int:
    """Material a piece on `square` would lose to the opponent's cheapest
    attacker if it were captured right now -- a one-ply SEE-lite over the
    current board (as opposed to failsafe.hangs_material, which evaluates a
    hypothetical move's destination). 0 if undefended-by-attackers, unattacked,
    or the king (never actually lost)."""
    piece = board.piece_at(square)
    if piece is None or piece.piece_type == chess.KING:
        return 0
    attackers = board.attackers(not piece.color, square)
    if not attackers:
        return 0
    piece_value = PIECE_VALUES[piece.piece_type]
    cheapest_attacker = min(PIECE_VALUES[board.piece_at(sq).piece_type] for sq in attackers)
    defenders = board.attackers(piece.color, square)
    if defenders:
        return max(0, piece_value - cheapest_attacker)
    return piece_value


def move_to_indices(move: chess.Move) -> tuple[int, int]:
    """(from_square, to_square), each 0-63 — matches the two 64-way policy heads."""
    return move.from_square, move.to_square
