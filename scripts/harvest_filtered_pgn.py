"""One-time harvest: stream Lichess monthly dumps, keep only the raw PGN text
(header + movetext, no parsing/encoding) of games that pass the rating-band
filter, and write one .pgn file per month -- optionally syncing each to
Google Drive as it finishes.

Why: build_dataset.py's slow part is downloading+decompressing+scanning the
full multi-GB monthly dump to find the ~3% of games that pass the filter;
encoding (chess.pgn parse + board_to_tensor) is comparatively fast. Once a
month's filtered games are harvested here, any future build_dataset.py run
(even with a changed encoding.py -- e.g. run5's channel-count change) can
re-encode from a small local/Drive .pgn file instead of going back to
Lichess. Run this locally, no GPU needed -- it's pure network+regex work,
reuses build_dataset.py's stream/filter functions unchanged.

Example:
    python scripts/harvest_filtered_pgn.py \\
        --months 2026-08 2026-07 2026-06 2026-05 2026-04 2026-03 2026-02 2026-01 2025-12 \\
        --out-dir data/filtered_pgn --drive-remote gdrive:chess_bot/filtered_pgn/
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from build_dataset import headers_pass_filter, iter_raw_games, open_pgn_stream, parse_headers_text  # noqa: E402


def month_source_url(month: str) -> str:
    return f"https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"


def harvest(source: str, out_path: Path, min_elo: int, max_elo: int, max_games: int | None, label: str = "") -> None:
    """Stream `source` (a database.lichess.org URL or a local .pgn/.pgn.zst
    path -- open_pgn_stream handles both), write the raw text of every game
    passing the rating-band filter to `out_path`. `label` is just for the
    progress log lines."""
    stream = open_pgn_stream(source)
    games_seen = games_kept = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for header_text, movetext_text in iter_raw_games(stream):
            games_seen += 1
            headers = parse_headers_text(header_text)
            if not headers_pass_filter(headers, min_elo, max_elo):
                continue
            f.write(header_text)
            f.write("\n")  # blank line between headers and movetext -- iter_raw_games needs it to
            f.write(movetext_text)  # tell the two blocks apart when this file is re-read later
            f.write("\n")  # blank line between games
            games_kept += 1
            if games_kept % 500 == 0:
                print(f"  {label}: {games_seen} seen, {games_kept} kept", file=sys.stderr)
            if max_games is not None and games_kept >= max_games:
                break
    print(f"{label}: done, {games_seen} seen, {games_kept} kept -> {out_path}", file=sys.stderr)


def sync_to_drive(path: Path, drive_remote: str) -> None:
    # Same graceful-failure pattern as train.py's save_checkpoint: a sync
    # hiccup shouldn't lose a month we already spent time downloading.
    # timeout=180 for the same reason train.py's rclone call has one: an
    # rclone hang here (real incident, run6, 2026-09-25) would otherwise
    # block this script indefinitely with no recovery.
    try:
        result = subprocess.run(["rclone", "copy", str(path), drive_remote], capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            print(f"  WARNING: rclone sync failed for {path.name}: {result.stderr.strip()[:300]}", file=sys.stderr)
        else:
            print(f"  synced {path.name} -> {drive_remote}", file=sys.stderr)
    except OSError as e:
        print(f"  WARNING: rclone sync failed to start for {path.name}: {e}", file=sys.stderr)
    except subprocess.TimeoutExpired:
        print(f"  WARNING: rclone sync timed out after 180s for {path.name}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--months", nargs="+", required=True, help="e.g. 2026-08 2026-07 ...")
    parser.add_argument("--out-dir", type=Path, default=Path("data/filtered_pgn"))
    parser.add_argument("--min-elo", type=int, default=2000)
    parser.add_argument("--max-elo", type=int, default=2200)
    parser.add_argument("--max-games", type=int, default=40_000, help="Cap per month -- matches build_dataset.py's default usage so far")
    parser.add_argument("--drive-remote", default=None, help="e.g. gdrive:chess_bot/filtered_pgn/ -- rclone-copy each month's file here as it finishes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for month in args.months:
        out_path = args.out_dir / f"{month}.pgn"
        harvest(month_source_url(month), out_path, args.min_elo, args.max_elo, args.max_games, label=month)
        if args.drive_remote:
            sync_to_drive(out_path, args.drive_remote)


if __name__ == "__main__":
    main()
