"""Cheap one-ply blunder check: does the ANN's proposed move hang a piece for
free? This is deliberately NOT a full static-exchange-evaluation engine and
does not catch multi-move tactics or positional errors -- only the single
most obvious class of blunder (a piece left on an undefended square, or an
under-defended one, where the opponent just takes it for nothing).
"""
import chess

_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,  # never actually lost; keeps it out of "cheap attacker" comparisons
}


def hanging_loss_at(board: chess.Board, square: int) -> int:
    """Material lost if the opponent captures the piece on `square` right
    now (one-ply SEE-lite): 0 if the square is empty, holds the king, is
    unattacked, or is defended well enough that the trade nets us nothing
    (floored at 0 -- this never credits a *winning* trade, only "safe" vs.
    "costs us N"). Used by hangs_material below to check every one of the
    mover's pieces, not just the one that moved.
    """
    piece = board.piece_at(square)
    if piece is None or piece.piece_type == chess.KING:
        return 0
    attackers = board.attackers(not piece.color, square)
    if not attackers:
        return 0
    piece_value = _PIECE_VALUES[piece.piece_type]
    cheapest_attacker = min(_PIECE_VALUES[board.piece_at(sq).piece_type] for sq in attackers)
    defenders = board.attackers(piece.color, square)
    if defenders:
        return max(0, piece_value - cheapest_attacker)
    return piece_value


def hangs_material(board: chess.Board, move: chess.Move) -> bool:
    """True if playing `move` loses material for free (a one-ply SEE-lite).

    Checks EVERY one of the mover's own pieces after the move, not just the
    one that moved -- a move can hang a *different* piece by removing its
    defender (e.g. a bishop blocking its own rook's file), which checking
    only the destination square misses entirely. Real incident (live
    Lichess game, 2026-09-30): ...Rc7 was defended by a rook on c1 through
    the empty c6 square; playing Bc6 (attacking a black rook on a8) looked
    safe for the bishop itself, but blocked that file and left the rook on
    c7 hanging to the black queen -- undetected until now.

    net = captured_value (what this move immediately wins, if a capture)
    minus the worst hanging_loss_at(...) among all of the mover's pieces
    after the move. Flagged only when net is negative, i.e. we come out
    behind overall. This still stops at one recapture level per piece (no
    deeper SEE, no multi-piece follow-up), per the plan's "one-ply is
    enough" scope -- it now just applies that same one-ply check board-wide
    instead of to a single square.
    """
    captured_value = _captured_value(board, move)

    after = board.copy(stack=False)
    mover = after.turn
    after.push(move)

    worst_loss = max(
        (hanging_loss_at(after, sq) for sq in chess.SQUARES if (p := after.piece_at(sq)) and p.color == mover),
        default=0,
    )
    return captured_value - worst_loss < 0


def _captured_value(board: chess.Board, move: chess.Move) -> int:
    """Value of whatever `move` captures on `board`, 0 if it's not a capture."""
    if board.is_en_passant(move):
        return _PIECE_VALUES[chess.PAWN]
    captured = board.piece_at(move.to_square)
    return _PIECE_VALUES[captured.piece_type] if captured is not None else 0


def pick_safe_move(board: chess.Board, candidates: list[chess.Move]) -> tuple[chess.Move, int]:
    """Walk the ANN's ranked candidate list (most-likely first) and return the
    first move that doesn't hang material for free.

    Returns (move, rejected_count):
    - rejected_count == 0: top candidate was safe, failsafe did not trigger.
    - rejected_count  > 0: that many higher-ranked candidates were rejected
      before finding a safe one -- caller can sum/count this across many
      calls to monitor how often the failsafe fires (Stage 5).
    - rejected_count == -1: NO candidate was safe. Refusing to move isn't an
      option, so the top-ranked candidate is returned anyway as a last
      resort -- -1 flags that this happened so the caller can distinguish it
      from an ordinary pass-through.
    """
    if not candidates:
        raise ValueError("candidates must be non-empty")

    for i, move in enumerate(candidates):
        if not hangs_material(board, move):
            return move, i

    return candidates[0], -1
