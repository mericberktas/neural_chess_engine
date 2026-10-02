"""Encode the harvested filtered PGNs to shard format v2 (boards + moves + legal
masks + attack bitboards) under data/v2/, for the run9 a/b/c comparison: the 16
train months run6 used plus the held-out test month. Local CPU only, no GPU.
Resumable: a finished month gets a .done marker and is skipped on re-run; a
month interrupted halfway is wiped and redone.

Runs from a SNAPSHOT of src/ taken at start, not the live checkout -- real
incident (2026-10-02): switching git branches while this ran made a month's
Python start hit "PermissionError: Access denied" on a src file git was
rewriting at that moment.

    .venv\\Scripts\\python.exe scripts\\encode_v2_local.py
    then: rclone copy data\\v2 gdrive:chess_bot/tensors/v2/ --transfers 8
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_MONTHS = [
    "2026-08", "2026-07", "2026-06", "2026-05", "2026-04", "2026-03", "2026-02", "2026-01",
    "2025-12", "2025-11", "2025-10", "2025-09", "2025-08", "2025-07", "2025-06", "2025-05",
]
TEST_MONTH = "2025-04"


def encode(snapshot: Path, month: str, out_dir: Path, split: str, max_games: int, workers: int) -> None:
    if (out_dir / ".done").exists():
        print(f"== {month}: already done, skipping", flush=True)
        return
    shutil.rmtree(out_dir, ignore_errors=True)
    print(f"== {month} ({split})", flush=True)
    result = subprocess.run(
        [sys.executable, "-u", str(snapshot / "build_dataset.py"),
         "--source", str(ROOT / "data" / "filtered_pgn" / f"{month}.pgn"), "--out-dir", str(out_dir),
         "--split", split, "--max-games", str(max_games), "--workers", str(workers)],
        capture_output=True, text=True,
    )
    lines = (result.stdout + result.stderr).strip().splitlines()
    print(lines[-1] if lines else "(no output)", flush=True)
    # build_dataset ends with os._exit(0), so judge success by the output, not the exit code
    if any(out_dir.glob("*/shard_00000.npz")):
        (out_dir / ".done").touch()
    else:
        print(f"!! {month} produced no shards", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--out-root", type=Path, default=ROOT / "data" / "v2")
    parser.add_argument("--months", nargs="+", default=TRAIN_MONTHS, help="train months (default: run6's 16)")
    parser.add_argument("--test-month", default=TEST_MONTH)
    parser.add_argument("--max-games", type=int, default=40_000, help="per train month; the test month uses min(this, 5000)")
    args = parser.parse_args()

    snapshot = Path(tempfile.mkdtemp(prefix="src_snapshot_"))
    shutil.copytree(ROOT / "src", snapshot, dirs_exist_ok=True)
    print(f"encoding from snapshot {snapshot}", flush=True)

    for month in args.months:
        encode(snapshot, month, args.out_root / month, "train", args.max_games, args.workers)
    encode(snapshot, args.test_month, args.out_root / f"test_{args.test_month}", "test", min(args.max_games, 5000), args.workers)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
