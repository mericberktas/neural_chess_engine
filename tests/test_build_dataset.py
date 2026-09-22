"""Assert-based self-check for src/build_dataset.py's fast-path game reader
and header pre-filter. No framework: python tests/test_build_dataset.py
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from build_dataset import (
    headers_pass_filter,
    iter_raw_games,
    parse_headers_text,
    _process_game,
)

# Three games back-to-back, Lichess dump layout: headers, blank line, one
# movetext line, blank line. Only the first should pass the Elo/Event filter.
SAMPLE_PGN = """[Event "Rated Blitz game"]
[Site "https://lichess.org/aaaaaaaa"]
[White "playerA"]
[Black "playerB"]
[Result "1-0"]
[WhiteElo "2100"]
[BlackElo "2050"]
[TimeControl "300+0"]

1. e4 { [%clk 0:05:00] } 1... e5 { [%clk 0:05:00] } 2. Nf3 { [%clk 0:04:58] } 1-0

[Event "Rated Bullet game"]
[Site "https://lichess.org/bbbbbbbb"]
[White "playerC"]
[Black "playerD"]
[Result "0-1"]
[WhiteElo "2100"]
[BlackElo "2050"]
[TimeControl "60+0"]

1. e4 { [%clk 0:01:00] } 1... e5 { [%clk 0:01:00] } 0-1

[Event "Rated Blitz game"]
[Site "https://lichess.org/cccccccc"]
[White "playerE"]
[Black "playerF"]
[Result "1/2-1/2"]
[WhiteElo "1500"]
[BlackElo "1500"]
[TimeControl "300+0"]

1. d4 { [%clk 0:05:00] } 1... d5 { [%clk 0:05:00] } 1/2-1/2

"""


def test_iter_raw_games_splits_three_games():
    games = list(iter_raw_games(io.StringIO(SAMPLE_PGN)))
    assert len(games) == 3
    for header_text, movetext_text in games:
        assert header_text.startswith("[Event ")
        assert movetext_text.strip()  # non-empty movetext


def test_header_filter_accepts_only_the_qualifying_game():
    games = list(iter_raw_games(io.StringIO(SAMPLE_PGN)))
    results = [headers_pass_filter(parse_headers_text(h), 2000, 2200) for h, _ in games]
    assert results == [True, False, False]  # blitz+in-band, bullet, out-of-band elo


def test_process_game_encodes_the_accepted_game_correctly():
    header_text, movetext_text = next(iter_raw_games(io.StringIO(SAMPLE_PGN)))
    target, results = _process_game((header_text, movetext_text, "train"), skip_plies=0, min_clock_seconds=0.0)
    assert target == "train"
    # e4, e5, Nf3 -> 3 positions, skip_plies=0 keeps all of them
    assert len(results) == 3
    first_board, first_move = results[0]
    assert first_board.shape == (18, 8, 8)
    assert first_move == (12, 28)  # e2e4


def test_process_game_respects_skip_plies_and_clock():
    header_text, movetext_text = next(iter_raw_games(io.StringIO(SAMPLE_PGN)))
    # skip_plies=2 drops e4/e5, keeps only Nf3
    target, results = _process_game((header_text, movetext_text, "train"), skip_plies=2, min_clock_seconds=0.0)
    assert len(results) == 1
    # min_clock_seconds above every recorded clock value -> nothing kept
    target, results = _process_game((header_text, movetext_text, "train"), skip_plies=0, min_clock_seconds=999.0)
    assert len(results) == 0


if __name__ == "__main__":
    test_iter_raw_games_splits_three_games()
    test_header_filter_accepts_only_the_qualifying_game()
    test_process_game_encodes_the_accepted_game_correctly()
    test_process_game_respects_skip_plies_and_clock()
    print("OK - all build_dataset checks passed")
