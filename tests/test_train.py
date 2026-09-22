"""Assert-based self-check for src/train.py's ShardShuffledSampler.
No framework: python tests/test_train.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from train import ShardDataset, ShardShuffledSampler

SHARD_SIZES = (5, 7, 3)  # deliberately uneven, and one tiny (size-3) shard


def _make_shards(shard_dir: Path) -> None:
    shard_dir.mkdir(parents=True)
    for i, size in enumerate(SHARD_SIZES):
        boards = np.zeros((size, 18, 8, 8), dtype=np.uint8)
        moves = np.arange(size * 2, dtype=np.int16).reshape(size, 2)
        np.savez(shard_dir / f"shard_{i:05d}.npz", boards=boards, moves=moves)


def test_sampler_is_a_valid_permutation_and_never_interleaves_shards():
    with tempfile.TemporaryDirectory() as tmp:
        shard_dir = Path(tmp) / "shards"
        _make_shards(shard_dir)
        dataset = ShardDataset(shard_dir)
        sampler = ShardShuffledSampler(dataset)

        assert len(sampler) == len(dataset) == sum(SHARD_SIZES)

        indices = list(sampler)
        assert sorted(indices) == list(range(len(dataset)))  # every index exactly once

        # Map each yielded index to its shard, and check indices are never
        # interleaved between shards: once a run leaves a shard, that shard
        # must not reappear later.
        def shard_of(idx: int) -> int:
            return int(np.searchsorted(dataset._offsets, idx, side="right") - 1)

        seen_shards = []
        for idx in indices:
            s = shard_of(idx)
            if not seen_shards or seen_shards[-1] != s:
                assert s not in seen_shards, "shard reappeared after being left -- interleaving detected"
                seen_shards.append(s)


def test_two_epochs_give_different_orders():
    # Not a strict guarantee (could coincide by chance), but with 15 items
    # across 3 shards the odds of an identical order twice are negligible --
    # this is here to catch an accidental fixed-seed regression.
    with tempfile.TemporaryDirectory() as tmp:
        shard_dir = Path(tmp) / "shards"
        _make_shards(shard_dir)
        dataset = ShardDataset(shard_dir)
        sampler = ShardShuffledSampler(dataset)
        order1 = list(sampler)
        order2 = list(sampler)
        assert order1 != order2


if __name__ == "__main__":
    test_sampler_is_a_valid_permutation_and_never_interleaves_shards()
    test_two_epochs_give_different_orders()
    print("OK - all train checks passed")
