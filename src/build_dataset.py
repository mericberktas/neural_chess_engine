"""Stream a Lichess PGN(.zst) dump, filter to a rating band, encode, and shard to disk.

Source can be a local .pgn / .pgn.zst file or a database.lichess.org URL — a URL
is streamed and decompressed on the fly, nothing is written to disk until a
qualifying position is found, so --max-games lets you smoke-test the whole
pipeline against a live remote dump without downloading the full month.

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
import io
import sys
import zlib
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import zstandard as zstd

from encoding import board_to_tensor, move_to_indices


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


def game_passes_filters(game: chess.pgn.Game, min_elo: int, max_elo: int) -> bool:
    if "Bullet" in game.headers.get("Event", ""):  # catches Bullet and UltraBullet
        return False
    try:
        white_elo = int(game.headers.get("WhiteElo", ""))
        black_elo = int(game.headers.get("BlackElo", ""))
    except ValueError:
        return False
    return min_elo <= white_elo <= max_elo and min_elo <= black_elo <= max_elo


def is_val_game(game_id: str, val_fraction: float) -> bool:
    # ponytail: crc32 bucket hash, deterministic per game id — good enough at hobby scale,
    # upgrade to a proper hash-partitioning scheme if the split ever needs to be reproduced
    # across languages/tools.
    bucket = zlib.crc32(game_id.encode("utf-8")) % 10_000
    return bucket < val_fraction * 10_000


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
    parser.add_argument("--max-games", type=int, default=None, help="Stop after this many qualifying games (local smoke test)")
    parser.add_argument("--shard-size", type=int, default=50_000)
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

    games_seen = games_kept = positions_kept = 0
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        games_seen += 1

        if not game_passes_filters(game, args.min_elo, args.max_elo):
            continue

        if args.split == "train":
            game_id = game.headers.get("Site", f"game-{games_seen}")
            writer = writers["val"] if is_val_game(game_id, args.val_fraction) else writers["train"]
        else:
            writer = writers["test"]

        board = game.board()
        added_any = False
        for ply, node in enumerate(game.mainline(), start=1):
            move = node.move
            if ply > args.skip_plies:
                clock = node.clock()
                if clock is None or clock >= args.min_clock_seconds:
                    writer.add(board_to_tensor(board), move_to_indices(move))
                    positions_kept += 1
                    added_any = True
            board.push(move)

        games_kept += added_any
        if games_kept and games_kept % 500 == 0:
            print(f"... {games_seen} games seen, {games_kept} kept, {positions_kept} positions", file=sys.stderr)

        if args.max_games is not None and games_kept >= args.max_games:
            break

    for writer in writers.values():
        writer.flush()

    print(f"Done: {games_seen} games seen, {games_kept} kept, {positions_kept} positions written to {args.out_dir}")


if __name__ == "__main__":
    main()
