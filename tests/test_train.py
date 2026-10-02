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
from train import (
    MODEL_ARG_NAMES, ShardDataset, ShardShuffledSampler, build_eval_set, evaluate, headline, legal_cross_entropy,
    save_checkpoint,
)

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
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max")
            args = argparse.Namespace(**{name: 1 for name in MODEL_ARG_NAMES}, drive_remote="gdrive:some/fake/remote/")
            path = Path(tmp) / "checkpoints" / "best.pt"
            save_checkpoint(path, model, optimizer, scheduler, args, step=1, top1=0.0, top3=0.0)  # must not raise
            assert path.exists()
    finally:
        train.subprocess.run = real_run


def test_save_checkpoint_survives_hung_rclone():
    # Regression test for a real incident (run6, 2026-09-25): rclone copy
    # hung indefinitely mid-upload with no timeout on the subprocess.run
    # call, silently blocking the entire training loop for 10+ minutes
    # until manually killed. subprocess.run is patched to simulate a
    # timeout rather than relying on an actually-hanging rclone.
    real_run = train.subprocess.run

    def fake_run(*a, **k):
        raise train.subprocess.TimeoutExpired(cmd=a[0], timeout=k.get("timeout"))

    train.subprocess.run = fake_run
    try:
        with tempfile.TemporaryDirectory() as tmp:
            model = nn.Linear(4, 4)
            optimizer = optim.AdamW(model.parameters())
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max")
            args = argparse.Namespace(**{name: 1 for name in MODEL_ARG_NAMES}, drive_remote="gdrive:some/fake/remote/")
            path = Path(tmp) / "checkpoints" / "best.pt"
            save_checkpoint(path, model, optimizer, scheduler, args, step=1, top1=0.0, top3=0.0)  # must not hang or raise
            assert path.exists()
    finally:
        train.subprocess.run = real_run


def test_resuming_restores_model_and_optimizer_state():
    # Exercises the same load_state_dict calls main()'s --resume-from block
    # makes, without needing to invoke the CLI end-to-end.
    with tempfile.TemporaryDirectory() as tmp:
        model = nn.Linear(4, 4)
        optimizer = optim.AdamW(model.parameters(), lr=1e-2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=1)
        # One real step so the optimizer actually has state (Adam's momentum
        # buffers) to resume, not just its freshly-initialized empty state.
        loss = model(torch.ones(1, 4)).sum()
        loss.backward()
        optimizer.step()

        args = argparse.Namespace(**{name: 1 for name in MODEL_ARG_NAMES}, drive_remote=None)
        path = Path(tmp) / "best.pt"
        save_checkpoint(path, model, optimizer, scheduler, args, step=123, top1=0.42, top3=0.7)

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
        assert "scheduler_state_dict" in ckpt


def test_resumed_scheduler_state_survives_a_second_construction():
    # Regression test for a real incident (run6, 2026-09-26): a freshly
    # constructed ReduceLROnPlateau starts its internal "best" at -inf, so
    # right after resuming it treated the next several val checks as
    # "improving" relative to -inf even though they were below the true
    # best_top1 -- delaying a real LR decay until early stopping (which
    # correctly compares against best_top1) fired first. Mirrors main()'s
    # --resume-from branch: if the checkpoint has scheduler_state_dict,
    # load it; a decayed LR and an elevated `best` must both survive.
    model = nn.Linear(4, 4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=1, factor=0.5)
    scheduler.step(0.40)  # improving
    scheduler.step(0.30)  # bad 1
    scheduler.step(0.30)  # bad 2 -> decays (patience=1 tolerates one bad check)
    decayed_lr = optimizer.param_groups[0]["lr"]
    assert decayed_lr == 5e-3  # confirms the decay above actually happened

    with tempfile.TemporaryDirectory() as tmp:
        args = argparse.Namespace(**{name: 1 for name in MODEL_ARG_NAMES}, drive_remote=None)
        path = Path(tmp) / "best.pt"
        save_checkpoint(path, model, optimizer, scheduler, args, step=1, top1=0.40, top3=0.6)
        ckpt = torch.load(path, map_location="cpu")

        fresh_model = nn.Linear(4, 4)
        fresh_optimizer = optim.AdamW(fresh_model.parameters(), lr=1e-2)
        fresh_optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        fresh_scheduler = optim.lr_scheduler.ReduceLROnPlateau(fresh_optimizer, mode="max", patience=1, factor=0.5)
        fresh_scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        assert fresh_optimizer.param_groups[0]["lr"] == decayed_lr
        assert fresh_scheduler.best == 0.40
        # A value between the two plateaued 0.30s and the true best 0.40 must
        # NOT look like an improvement to the resumed scheduler (it would to
        # a freshly-constructed one, whose best starts at -inf).
        fresh_scheduler.step(0.35)
        assert fresh_scheduler.num_bad_epochs == 1


def test_scheduler_best_is_seeded_for_a_checkpoint_without_scheduler_state():
    # The fallback path in main()'s resume block for a checkpoint saved
    # before this fix existed (no "scheduler_state_dict" key): seed
    # `.best` with the true best_top1 instead of leaving a fresh
    # scheduler's -inf, so it doesn't mistake a below-best value for an
    # improvement right after resuming.
    model = nn.Linear(4, 4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=1)
    best_top1 = 0.4871
    scheduler.best = best_top1  # the fallback main() applies when resuming an old-format checkpoint
    scheduler.step(0.45)  # below best_top1 -- must count as a bad check, not an improvement
    assert scheduler.num_bad_epochs == 1


def _make_v2_shards(shard_dir: Path) -> None:
    """Random but fixed-seed shard-format-v2 shards (boards, moves, legal, attacks)."""
    shard_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i, size in enumerate(SHARD_SIZES):
        moves = rng.integers(0, 64, size=(size, 2)).astype(np.int16)
        legal = rng.integers(0, 256, size=(size, 512), dtype=np.uint8)
        pair = moves[:, 0].astype(np.int64) * 64 + moves[:, 1]
        legal[np.arange(size), pair // 8] |= (1 << (pair % 8)).astype(np.uint8)  # the played move is always legal
        np.savez(
            shard_dir / f"shard_{i:05d}.npz",
            boards=rng.integers(0, 2, size=(size, NUM_CHANNELS, 8, 8), dtype=np.uint8),
            moves=moves, legal=legal,
            attacks=rng.integers(0, 256, size=(size, 64, 8), dtype=np.uint8),
        )


def test_getitems_matches_per_sample_getitem_in_callers_order():
    with tempfile.TemporaryDirectory() as tmp:
        shard_dir = Path(tmp) / "shards"
        _make_v2_shards(shard_dir)
        dataset = ShardDataset(shard_dir)
        assert dataset.has_extras
        indices = [14, 0, 7, 3, 12, 5, 7]  # all three shards, unsorted, one repeat
        boards, moves, legal, attacks = dataset.__getitems__(indices)
        assert boards.dtype == torch.uint8 and moves.dtype == torch.int64
        for row, idx in enumerate(indices):
            expected_board, expected_move = dataset[idx]
            assert torch.equal(boards[row].float(), expected_board)
            assert torch.equal(moves[row], expected_move)
        # legal/attacks rows come from the right shard + local position
        shard_files = sorted(shard_dir.glob("shard_*.npz"))
        with np.load(shard_files[2]) as d:  # global idx 14 = shard 2 (offset 12), local 2
            assert np.array_equal(legal[0].numpy(), d["legal"][2])
            assert np.array_equal(attacks[0].numpy(), d["attacks"][2])


def test_build_eval_set_is_a_strided_sample_across_every_shard():
    with tempfile.TemporaryDirectory() as tmp:
        shard_dir = Path(tmp) / "shards"
        _make_v2_shards(shard_dir)
        dataset = ShardDataset(shard_dir)
        eval_set = build_eval_set(dataset, num_positions=5)
        expected_idx = [0, 3, 7, 10, 14]  # linspace(0, 14, 5): hits shard 0 (0-4), 1 (5-11) and 2 (12-14)
        expected = dataset.__getitems__(expected_idx)
        for got, want in zip(eval_set, expected):
            assert torch.equal(got, want)
        # asking for more than exist just returns everything once
        assert len(build_eval_set(dataset, num_positions=10_000)[1]) == len(dataset)


def test_mixed_shard_formats_are_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        v1, v2 = Path(tmp) / "v1", Path(tmp) / "v2"
        _make_shards(v1)
        _make_v2_shards(v2)
        try:
            ShardDataset([v1, v2])
        except ValueError:
            return
        raise AssertionError("mixing v1 and v2 shards should raise")


class _FixedScoreModel(nn.Module):
    """Always scores (0,1) highest, then (2,3), then everything else equally."""

    def joint_logits(self, boards):
        joint = torch.zeros(boards.shape[0], 64, 64)
        joint[:, 0, 1] = 10.0
        joint[:, 2, 3] = 5.0
        return joint


def test_evaluate_legal_masking_changes_the_headline_metric():
    n = 4
    legal = np.zeros((n, 64 * 64), dtype=bool)
    legal[:, 2 * 64 + 3] = True  # only the true move and one other are legal; (0,1) is NOT
    legal[:, 4 * 64 + 5] = True
    eval_set = (
        torch.zeros(n, NUM_CHANNELS, 8, 8, dtype=torch.uint8),
        torch.tensor([[2, 3]] * n),
        torch.from_numpy(np.packbits(legal, axis=1, bitorder="little")),
        None,
    )
    metrics = evaluate(_FixedScoreModel(), eval_set, torch.device("cpu"))
    assert metrics["top1"] == 0.0  # unmasked argmax is the illegal (0,1)
    assert metrics["legal_top1"] == 1.0  # masked argmax is the true move
    assert headline(metrics) == (1.0, 1.0)
    # no legal masks (shard v1) -> unmasked numbers only, headline falls back to them
    metrics_v1 = evaluate(_FixedScoreModel(), (eval_set[0], eval_set[1], None, None), torch.device("cpu"))
    assert "legal_top1" not in metrics_v1 and headline(metrics_v1) == (0.0, 1.0)


def test_main_trains_end_to_end_on_v2_shards_and_checkpoint_reloads():
    with tempfile.TemporaryDirectory() as tmp:
        shard_dir, out_dir = Path(tmp) / "shards", Path(tmp) / "out"
        _make_v2_shards(shard_dir)
        argv = [
            "train.py", "--train-dir", str(shard_dir), "--val-dir", str(shard_dir), "--out-dir", str(out_dir),
            "--batch-size", "4", "--d-model", "16", "--nhead", "2", "--num-layers", "2", "--dim-feedforward", "32",
            "--gab-per-layer", "--val-interval", "3", "--val-positions", "8", "--max-steps", "6", "--device", "cpu",
        ]
        old_argv, sys.argv = sys.argv, argv
        try:
            train.main()
        finally:
            sys.argv = old_argv
        ckpt = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
        assert ckpt["model_args"]["gab_per_layer"] is True
        from model import ChessTransformer
        model = ChessTransformer(**ckpt["model_args"])
        model.load_state_dict(ckpt["model_state_dict"])


def test_legal_cross_entropy_matches_softmax_over_the_legal_subset():
    torch.manual_seed(0)
    joint = torch.randn(3, 4096, requires_grad=True)
    legal = torch.zeros(3, 4096, dtype=torch.bool)
    targets = torch.tensor([5, 100, 4095])
    for row, extra in enumerate([[7, 9, 11], [1, 2], [3000, 17, 18, 19]]):
        legal[row, targets[row]] = True
        legal[row, extra] = True
    for smoothing in (0.0, 0.1):
        losses = []
        for row in range(3):
            idx = legal[row].nonzero().squeeze(1)
            logp = torch.log_softmax(joint[row, idx], dim=0)
            nll = -logp[(idx == targets[row]).nonzero().item()]
            losses.append((1 - smoothing) * nll + smoothing * (-logp.mean()))
        expected = torch.stack(losses).mean()
        got = legal_cross_entropy(joint, targets, legal, smoothing)
        assert torch.isfinite(got) and torch.allclose(got, expected, atol=1e-5)
    got.backward()
    assert torch.isfinite(joint.grad).all()
    assert (joint.grad[~legal] == 0).all()  # illegal pairs get no gradient


def test_pair_head_without_legal_loss_is_rejected_and_the_combination_trains():
    with tempfile.TemporaryDirectory() as tmp:
        shard_dir, out_dir = Path(tmp) / "shards", Path(tmp) / "out"
        _make_v2_shards(shard_dir)
        base = [
            "train.py", "--train-dir", str(shard_dir), "--val-dir", str(shard_dir), "--out-dir", str(out_dir),
            "--batch-size", "4", "--d-model", "16", "--nhead", "2", "--num-layers", "2", "--dim-feedforward", "32",
            "--val-interval", "3", "--val-positions", "8", "--max-steps", "6", "--device", "cpu",
        ]
        old_argv = sys.argv
        try:
            sys.argv = base + ["--pair-head"]
            try:
                train.parse_args()
            except SystemExit:
                pass
            else:
                raise AssertionError("--pair-head without --legal-loss should exit")
            sys.argv = base + ["--pair-head", "--legal-loss", "--gab-per-layer"]
            train.main()
        finally:
            sys.argv = old_argv
        ckpt = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
        assert ckpt["model_args"]["pair_head"] is True
        assert "pair_q.weight" in ckpt["model_state_dict"]


if __name__ == "__main__":
    test_sampler_is_a_valid_permutation_and_never_interleaves_shards()
    test_two_epochs_give_different_orders()
    test_shard_dataset_combines_multiple_directories()
    test_save_checkpoint_survives_missing_rclone()
    test_save_checkpoint_survives_hung_rclone()
    test_resuming_restores_model_and_optimizer_state()
    test_resumed_scheduler_state_survives_a_second_construction()
    test_scheduler_best_is_seeded_for_a_checkpoint_without_scheduler_state()
    test_getitems_matches_per_sample_getitem_in_callers_order()
    test_build_eval_set_is_a_strided_sample_across_every_shard()
    test_mixed_shard_formats_are_rejected()
    test_evaluate_legal_masking_changes_the_headline_metric()
    test_main_trains_end_to_end_on_v2_shards_and_checkpoint_reloads()
    test_legal_cross_entropy_matches_softmax_over_the_legal_subset()
    test_pair_head_without_legal_loss_is_rejected_and_the_combination_trains()
    print("OK - all train checks passed")
