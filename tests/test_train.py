"""Assert-based self-check for src/train.py's ShardShuffledSampler and the
checkpoint drive-sync failure handling. No framework: python tests/test_train.py
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import train
from encoding import NUM_CHANNELS
from train import MODEL_ARG_NAMES, ShardDataset, ShardShuffledSampler, save_checkpoint

SHARD_SIZES = (5, 7, 3)  # deliberately uneven, and one tiny (size-3) shard


def _make_shards(shard_dir: Path) -> None:
    shard_dir.mkdir(parents=True)
    for i, size in enumerate(SHARD_SIZES):
        boards = np.zeros((size, NUM_CHANNELS, 8, 8), dtype=np.uint8)
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
            optimizer = optim.AdamW(model.parameters())
            args = argparse.Namespace(**{name: 1 for name in MODEL_ARG_NAMES}, drive_remote="gdrive:some/fake/remote/")
            path = Path(tmp) / "checkpoints" / "best.pt"
            save_checkpoint(path, model, optimizer, args, step=1, top1=0.0, top3=0.0)  # must not raise
            assert path.exists()
    finally:
        train.subprocess.run = real_run


def test_resuming_restores_model_and_optimizer_state():
    # Exercises the same load_state_dict calls main()'s --resume-from block
    # makes, without needing to invoke the CLI end-to-end.
    with tempfile.TemporaryDirectory() as tmp:
        model = nn.Linear(4, 4)
        optimizer = optim.AdamW(model.parameters(), lr=1e-2)
        # One real step so the optimizer actually has state (Adam's momentum
        # buffers) to resume, not just its freshly-initialized empty state.
        loss = model(torch.ones(1, 4)).sum()
        loss.backward()
        optimizer.step()

        args = argparse.Namespace(**{name: 1 for name in MODEL_ARG_NAMES}, drive_remote=None)
        path = Path(tmp) / "best.pt"
        save_checkpoint(path, model, optimizer, args, step=123, top1=0.42, top3=0.7)

        fresh_model = nn.Linear(4, 4)
        fresh_optimizer = optim.AdamW(fresh_model.parameters(), lr=1e-2)
        assert not torch.equal(fresh_model.weight, model.weight)  # different random init

        ckpt = torch.load(path, map_location="cpu")
        fresh_model.load_state_dict(ckpt["model_state_dict"])
        fresh_optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        assert torch.equal(fresh_model.weight, model.weight)
        assert ckpt["step"] == 123
        assert ckpt["val_top1"] == 0.42
        # Adam's per-parameter step count is real optimizer state, not just
        # weights -- confirms load_state_dict actually restored it.
        resumed_state = next(iter(fresh_optimizer.state_dict()["state"].values()))
        assert resumed_state["step"] == 1


if __name__ == "__main__":
    test_sampler_is_a_valid_permutation_and_never_interleaves_shards()
    test_two_epochs_give_different_orders()
    test_shard_dataset_combines_multiple_directories()
    test_save_checkpoint_survives_missing_rclone()
    test_resuming_restores_model_and_optimizer_state()
    print("OK - all train checks passed")
