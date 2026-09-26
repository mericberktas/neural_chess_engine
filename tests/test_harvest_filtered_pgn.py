"""Assert-based self-check for scripts/harvest_filtered_pgn.py. No framework:
python tests/test_harvest_filtered_pgn.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import harvest_filtered_pgn
from build_dataset import iter_raw_games
from harvest_filtered_pgn import harvest, sync_to_drive

# Same three-game fixture as tests/test_build_dataset.py: only the first game
# (blitz, both players in-band) should survive the filter.
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


def test_harvest_writes_only_the_qualifying_game_verbatim():
    with tempfile.TemporaryDirectory() as tmp:
        source_path = Path(tmp) / "source.pgn"
        source_path.write_text(SAMPLE_PGN, encoding="utf-8")
        out_path = Path(tmp) / "harvested.pgn"

        harvest(str(source_path), out_path, min_elo=2000, max_elo=2200, max_games=None, label="test")

        harvested = out_path.read_text(encoding="utf-8")
        assert 'Site "https://lichess.org/aaaaaaaa"' in harvested
        assert "https://lichess.org/bbbbbbbb" not in harvested  # bullet, rejected
        assert "https://lichess.org/cccccccc" not in harvested  # out-of-band elo, rejected
        assert "1. e4" in harvested and "2. Nf3" in harvested


def test_harvest_respects_max_games():
    # Two qualifying games this time, capped at 1.
    two_qualifying = SAMPLE_PGN.replace(
        '[Event "Rated Bullet game"]', '[Event "Rated Blitz game"]',
    ).replace('[TimeControl "60+0"]', '[TimeControl "300+0"]')
    with tempfile.TemporaryDirectory() as tmp:
        source_path = Path(tmp) / "source.pgn"
        source_path.write_text(two_qualifying, encoding="utf-8")
        out_path = Path(tmp) / "harvested.pgn"

        harvest(str(source_path), out_path, min_elo=2000, max_elo=2200, max_games=1, label="test")

        harvested = out_path.read_text(encoding="utf-8")
        assert "https://lichess.org/aaaaaaaa" in harvested
        assert "https://lichess.org/bbbbbbbb" not in harvested  # would've qualified too, but cap=1


def test_harvested_file_is_correctly_re_splittable_into_separate_games():
    # Regression test: an earlier version wrote header_text + movetext_text
    # with no blank line between them, so re-reading the file with
    # iter_raw_games (exactly what build_dataset.py does with it later)
    # merged pairs of games into one -- 50 harvested games silently became
    # ~25 on the next read. Two qualifying games in, two must come back out.
    two_qualifying = SAMPLE_PGN.replace(
        '[Event "Rated Bullet game"]', '[Event "Rated Blitz game"]',
    ).replace('[TimeControl "60+0"]', '[TimeControl "300+0"]')
    with tempfile.TemporaryDirectory() as tmp:
        source_path = Path(tmp) / "source.pgn"
        source_path.write_text(two_qualifying, encoding="utf-8")
        out_path = Path(tmp) / "harvested.pgn"

        harvest(str(source_path), out_path, min_elo=2000, max_elo=2200, max_games=None, label="test")

        with open(out_path, encoding="utf-8") as f:
            re_split = list(iter_raw_games(f))
        assert len(re_split) == 2
        assert "https://lichess.org/aaaaaaaa" in re_split[0][0]
        assert "https://lichess.org/bbbbbbbb" in re_split[1][0]
        assert "1. e4" in re_split[0][1]


def test_sync_to_drive_survives_hung_rclone():
    # Same class of real incident as test_train.py's
    # test_save_checkpoint_survives_hung_rclone (run6, 2026-09-25): rclone
    # copy hung indefinitely with no timeout, blocking the caller forever.
    real_run = harvest_filtered_pgn.subprocess.run

    def fake_run(*a, **k):
        raise harvest_filtered_pgn.subprocess.TimeoutExpired(cmd=a[0], timeout=k.get("timeout"))

    harvest_filtered_pgn.subprocess.run = fake_run
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "2026-08.pgn"
            path.write_text("dummy", encoding="utf-8")
            sync_to_drive(path, "gdrive:some/fake/remote/")  # must not hang or raise
    finally:
        harvest_filtered_pgn.subprocess.run = real_run


if __name__ == "__main__":
    test_harvest_writes_only_the_qualifying_game_verbatim()
    test_harvest_respects_max_games()
    test_harvested_file_is_correctly_re_splittable_into_separate_games()
    test_sync_to_drive_survives_hung_rclone()
    print("OK - all harvest_filtered_pgn checks passed")
