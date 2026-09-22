"""Thin wrapper around chess.syzygy for endgame perfect play.

When the board has few enough pieces, the ANN is bypassed entirely and the
Syzygy-recommended move is played instead. Missing table files (only a
handful of 3-5 piece files are kept locally, not the full ~1GB set) must not
crash the engine — `best_move` returns None so the caller falls back to ANN.
"""
import chess
import chess.syzygy

DEFAULT_PIECE_THRESHOLD = 5

open_tablebase = chess.syzygy.open_tablebase


def should_probe(board: chess.Board, piece_threshold: int = DEFAULT_PIECE_THRESHOLD) -> bool:
    return chess.popcount(board.occupied) <= piece_threshold


def best_move(board: chess.Board, tablebase: chess.syzygy.Tablebase) -> chess.Move | None:
    """DTZ-optimal move for `board`, or None if the position/table isn't
    available locally (missing table file, castling rights still held, too
    many pieces, ...) — caller should fall back to the ANN in that case.

    Ranks candidate moves by (our WDL class, opponent's DTZ) lexicographically:
    WDL class alone (win > draw > loss) decides correctness — comparing raw
    DTZ numbers across different WDL classes is unsafe because "cursed win"
    /"blessed loss" DTZ values are encoded past +-100, outside the plain
    win/loss range. DTZ only breaks ties inside a single WDL class, to make
    fastest progress toward a win (or slowest toward a loss).
    """
    best, best_key = None, None
    for move in board.legal_moves:
        board.push(move)
        try:
            if board.is_checkmate():
                key = (2, 0)  # just delivered mate: best possible, beats any other win
            else:
                opponent_wdl = tablebase.probe_wdl(board)
                opponent_dtz = tablebase.probe_dtz(board)
                key = (-opponent_wdl, opponent_dtz)
        except KeyError:
            return None
        finally:
            board.pop()
        if best_key is None or key > best_key:
            best, best_key = move, key
    return best
