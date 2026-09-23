"""Assert-based self-check for src/train.py's ShardShuffledSampler and the
checkpoint drive-sync failure handling. No framework: python tests/test_train.py
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import train
from train import MODEL_ARG_NAMES, ShardDataset, ShardShuffledSampler, save_checkpoint

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


def test_shard_dataset_combines_multiple_directories():
    with tempfile.TemporaryDirectory() as tmp:
        dir_a, dir_b = Path(tmp) / "a", Path(tmp) / "b"
        _make_shards(dir_a)  # 5+7+3 = 15 items
        _make_shards(dir_b)  # another 15, independently numbered shard_00000..02
        combined = ShardDataset([dir_a, dir_b])
        assert len(combined) == 2 * sum(SHARD_SIZES)
        assert len(combined.files) == 6  # 3 shards from each dir, no collision

        single = ShardDataset(dir_a)  # a bare Path still works, not just a list
        assert len(single) == sum(SHARD_SIZES)


def test_save_checkpoint_survives_missing_rclone():
    # Regression test for a real bug: subprocess.run() raises
    # FileNotFoundError for a missing executable, which a returncode-only
    # check doesn't catch. subprocess.run is patched directly rather than
    # relying on rclone actually being absent from PATH -- on a machine
    # where rclone *is* installed and configured (this one, later), the
    # unpatched version of this test really hit the network and wrote a
    # junk checkpoint to a real Google Drive remote. Tests must not depend
    # on ambient environment state for something this consequential.
    real_run = train.subprocess.run
    train.subprocess.run = lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("simulated: rclone not found"))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            model = nn.Linear(4, 4)
            args = argparse.Namespace(**{name: 1 for name in MODEL_ARG_NAMES}, drive_remote="gdrive:some/fake/remote/")
            path = Path(tmp) / "checkpoints" / "best.pt"
            save_checkpoint(path, model, args, step=1, top1=0.0, top3=0.0)  # must not raise
            assert path.exists()
    finally:
        train.subprocess.run = real_run


if __name__ == "__main__":
    test_sampler_is_a_valid_permutation_and_never_interleaves_shards()
    test_two_epochs_give_different_orders()
    test_shard_dataset_combines_multiple_directories()
    test_save_checkpoint_survives_missing_rclone()
    print("OK - all train checks passed")
