"""Cheap one-ply blunder check: does the ANN's proposed move hang a piece for
free? This is deliberately NOT a full static-exchange-evaluation engine and
does not catch multi-move tactics or positional errors -- only the single
most obvious class of blunder (moving a piece to an undefended square, or an
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


def hangs_material(board: chess.Board, move: chess.Move) -> bool:
    """True if playing `move` loses material for free (a one-ply SEE-lite).

    Looks only at the piece that just moved, on its destination square:
    - What did this move capture, if anything (`captured_value`)?
    - Is the destination now attacked by the opponent? If not, done, safe.
    - If it's undefended by us, the opponent captures it outright -- net is
      `captured_value - piece_value`.
    - If it's defended, the opponent will still grab it with their cheapest
      attacker when that attacker is worth less than our piece, and we
      recapture that attacker -- net is
      `captured_value - piece_value + cheapest_attacker`.
    Flagged only when that net is negative, i.e. we come out behind. This
    correctly leaves ordinary trades/recaptures (net ~0) and winning
    captures (net > 0) unflagged -- only genuine "gave up material and got
    nothing/not enough back" moves trip it. It stops at one recapture level
    (no deeper SEE), per the plan's "one-ply is enough" scope.
    """
    captured_value = _captured_value(board, move)

    after = board.copy(stack=False)
    mover = after.turn
    after.push(move)

    piece = after.piece_at(move.to_square)
    if piece is None:
        return False

    opponent = not mover
    attackers = after.attackers(opponent, move.to_square)
    if not attackers:
        return False

    piece_value = _PIECE_VALUES[piece.piece_type]
    cheapest_attacker = min(_PIECE_VALUES[after.piece_at(sq).piece_type] for sq in attackers)

    defenders = after.attackers(mover, move.to_square)
    recapture_value = cheapest_attacker if defenders else 0

    net = captured_value - piece_value + recapture_value
    return net < 0


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
