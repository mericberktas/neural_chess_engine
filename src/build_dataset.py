"""Stream a Lichess PGN(.zst) dump, filter to a rating band, encode, and shard to disk.

Source can be a local .pgn / .pgn.zst file or a database.lichess.org URL — a URL
is streamed and decompressed on the fly, nothing is written to disk until a
qualifying position is found, so --max-games lets you smoke-test the whole
pipeline against a live remote dump without downloading the full month.

~97% of games get rejected by the Elo/Event filter, so the reader splits each
game into raw (header, movetext) text first and only hands the ~3% that pass
a cheap header check to chess.pgn's real parser -- skips board-simulation
cost for everything that was going to be thrown away anyway. That parse-and-
encode step for accepted games is spread across a worker pool (--workers)
since it's the remaining CPU-heavy part and most cores otherwise sit idle.

Examples:
    # local smoke test, no download: read straight from the remote stream, stop early
    python src/build_dataset.py \\
        --source https://database.lichess.org/standard/lichess_db_standard_rated_2026-08.pgn.zst \\
        --out-dir data/smoke --max-games 2000

    # full run against an already-downloaded month (e.g. on a cloud box)
    python src/build_dataset.py --source lichess_db_standard_rated_2026-08.pgn.zst --out-dir data/train_2026-08

    # a separate month held out entirely as the test set
    python src/build_dataset.py --source lichess_db_standard_rated_2026-04.pgn.zst --out-dir data/test_2026-04 --split test
"""
import argparse
import functools
import io
import multiprocessing
import os
import re
import sys
import zlib
from pathlib import Path

import chess.pgn
import numpy as np
import zstandard as zstd

from encoding import board_to_tensor, move_to_indices

_HEADER_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$', re.MULTILINE)


def open_pgn_stream(source: str):
    if source.startswith("http://") or source.startswith("https://"):
        import requests

        resp = requests.get(source, stream=True)
        resp.raise_for_status()
        raw = resp.raw
    elif source.endswith(".zst"):
        raw = open(source, "rb")
    else:
        return open(source, "r", encoding="utf-8", errors="replace")

    reader = zstd.ZstdDecompressor().stream_reader(raw)
    return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")


def iter_raw_games(stream):
    """Yield (header_text, movetext_text) per game as plain text, with no
    parse-tree construction -- cheap enough to run on every game so a
    header-only filter can reject most of them without ever invoking
    chess.pgn.read_game(). Matches Lichess's dump layout: N header lines,
    blank line, movetext, blank line, repeat.
    """
    header_lines: list[str] = []
    movetext_lines: list[str] = []
    in_movetext = False
    for line in stream:
        if line.strip():
            (movetext_lines if in_movetext else header_lines).append(line)
        elif header_lines and not in_movetext:
            in_movetext = True
        elif in_movetext and movetext_lines:
            yield "".join(header_lines), "".join(movetext_lines)
            header_lines, movetext_lines, in_movetext = [], [], False
    if header_lines and movetext_lines:
        yield "".join(header_lines), "".join(movetext_lines)


def parse_headers_text(header_text: str) -> dict[str, str]:
    return dict(_HEADER_RE.findall(header_text))


def headers_pass_filter(headers, min_elo: int, max_elo: int) -> bool:
    if "Bullet" in headers.get("Event", ""):  # catches Bullet and UltraBullet
        return False
    try:
        white_elo = int(headers.get("WhiteElo", ""))
        black_elo = int(headers.get("BlackElo", ""))
    except ValueError:
        return False
    return min_elo <= white_elo <= max_elo and min_elo <= black_elo <= max_elo


def is_val_game(game_id: str, val_fraction: float) -> bool:
    # ponytail: crc32 bucket hash, deterministic per game id — good enough at hobby scale,
    # upgrade to a proper hash-partitioning scheme if the split ever needs to be reproduced
    # across languages/tools.
    bucket = zlib.crc32(game_id.encode("utf-8")) % 10_000
    return bucket < val_fraction * 10_000


def _process_game(item: tuple[str, str, str], skip_plies: int, min_clock_seconds: float):
    """Runs in a worker process: full-parse one already-accepted game and
    encode its (board, move) pairs. This is the expensive step (PGN tree
    build + board simulation per move), only ever called on the ~3% of
    games that passed the cheap header filter.
    """
    header_text, movetext_text, target = item
    game = chess.pgn.read_game(io.StringIO(header_text + "\n" + movetext_text))
    if game is None:
        return target, []
    board = game.board()
    results = []
    for ply, node in enumerate(game.mainline(), start=1):
        move = node.move
        if ply > skip_plies:
            clock = node.clock()
            if clock is None or clock >= min_clock_seconds:
                results.append((board_to_tensor(board), move_to_indices(move)))
        board.push(move)
    return target, results


class ShardWriter:
    def __init__(self, out_dir: Path, shard_size: int):
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.boards: list[np.ndarray] = []
        self.moves: list[tuple[int, int]] = []
        self.shard_index = 0
        out_dir.mkdir(parents=True, exist_ok=True)

    def add(self, board_tensor: np.ndarray, move_idx: tuple[int, int]) -> None:
        self.boards.append(board_tensor)
        self.moves.append(move_idx)
        if len(self.boards) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.boards:
            return
        path = self.out_dir / f"shard_{self.shard_index:05d}.npz"
        np.savez_compressed(
            path,
            boards=np.stack(self.boards),
            moves=np.array(self.moves, dtype=np.int16),
        )
        self.shard_index += 1
        self.boards.clear()
        self.moves.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Local .pgn/.pgn.zst path or a database.lichess.org URL")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--split", choices=["train", "test"], default="train",
        help="'train' also carves out a val/ slice by game id; 'test' writes everything as-is "
             "(point it at a separate month's dump so the test set is time-disjoint from training)",
    )
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--min-elo", type=int, default=2000)
    parser.add_argument("--max-elo", type=int, default=2200)
    parser.add_argument("--skip-plies", type=int, default=10, help="Half-moves skipped at the start of each game (book/opening)")
    parser.add_argument("--min-clock-seconds", type=float, default=30.0)
    parser.add_argument("--max-games", type=int, default=None, help="Stop after roughly this many qualifying games")
    parser.add_argument("--shard-size", type=int, default=50_000)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
        help="Worker processes for parsing/encoding accepted games (the stream itself is read single-threaded)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.split == "train":
        writers = {
            "train": ShardWriter(args.out_dir / "train", args.shard_size),
            "val": ShardWriter(args.out_dir / "val", args.shard_size),
        }
    else:
        writers = {"test": ShardWriter(args.out_dir / "test", args.shard_size)}

    stream = open_pgn_stream(args.source)
    games_seen = 0

    def accepted_games():
        nonlocal games_seen
        for header_text, movetext_text in iter_raw_games(stream):
            games_seen += 1
            headers = parse_headers_text(header_text)
            if not headers_pass_filter(headers, args.min_elo, args.max_elo):
                continue
            if args.split == "train":
                game_id = headers.get("Site", f"game-{games_seen}")
                target = "val" if is_val_game(game_id, args.val_fraction) else "train"
            else:
                target = "test"
            yield header_text, movetext_text, target

    worker = functools.partial(_process_game, skip_plies=args.skip_plies, min_clock_seconds=args.min_clock_seconds)

    games_kept = positions_kept = 0
    # Not `with multiprocessing.Pool(...)`: that context manager's __exit__
    # calls pool.join(), which can hang indefinitely when --max-games stops
    # consumption early -- a known multiprocessing gotcha where prefetched,
    # already-dispatched-but-unconsumed tasks (chunksize=16, so several
    # batches ahead) deadlock the pool's internal feeder thread on cleanup.
    # Observed live twice (2026-09-23) and previously worked around
    # externally with a `timeout + pkill` wrapper in runpod_overnight.sh --
    # fixed at the source here instead: terminate() (sends SIGTERM, does not
    # block) and skip join entirely.
    pool = multiprocessing.Pool(args.workers)
    try:
        for target, results in pool.imap_unordered(worker, accepted_games(), chunksize=16):
            if not results:
                continue
            writer = writers[target]
            for board_tensor, move_idx in results:
                writer.add(board_tensor, move_idx)
                positions_kept += 1
            games_kept += 1

            if games_kept % 500 == 0:
                print(f"... {games_seen} games seen, {games_kept} kept, {positions_kept} positions", file=sys.stderr)

            if args.max_games is not None and games_kept >= args.max_games:
                break
    finally:
        pool.terminate()

    for writer in writers.values():
        writer.flush()

    print(f"Done: {games_seen} games seen, {games_kept} kept, {positions_kept} positions written to {args.out_dir}")
    # Python's normal interpreter shutdown joins any remaining multiprocessing
    # children (registered via atexit) -- exactly the hang terminate() above
    # is meant to avoid. os._exit() skips atexit entirely; nothing after
    # main() needs to run, so this is safe.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
