"""Sanity-check shard format v2 directories: every position's played move must
be set in that position's own legal-move mask (a free end-to-end consistency
check between the encoder, the writer and the mask bit layout), and the
attack bitboards must have the expected shape.

    python scripts/verify_shards.py data/v2/2026-08/train data/v2/2026-08/val
"""
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    total = 0
    for shard_dir in sys.argv[1:]:
        for f in sorted(Path(shard_dir).glob("shard_*.npz")):
            with np.load(f) as d:
                moves, legal, attacks = d["moves"].astype(np.int64), d["legal"], d["attacks"]
            assert legal.shape == (len(moves), 512) and attacks.shape == (len(moves), 64, 8), f
            bits = np.unpackbits(legal, axis=1, bitorder="little")
            played = bits[np.arange(len(moves)), moves[:, 0] * 64 + moves[:, 1]]
            assert played.all(), f"{f}: {int((played == 0).sum())} positions whose played move is not in their legal mask"
            total += len(moves)
    print(f"OK - {total} positions, every played move is in its legal mask")


if __name__ == "__main__":
    main()
