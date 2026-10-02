"""Train the ChessTransformer policy network on build_dataset.py shards.

Reads train/ and val/ subfolders of shard_*.npz files (boards: (N,NUM_CHANNELS,8,8)
uint8, moves: (N,2) int16 = from_square, to_square; shard format v2 also has
legal: (N,512) uint8 and attacks: (N,64,8) uint8 -- see encoding.position_extras).
Val is evaluated every --val-interval steps (not once per epoch, since one epoch
over the real dataset can be huge) and the checkpoint with the best val top-1 is
kept. Val runs on a FIXED evenly-strided sample of --val-positions positions
drawn from every val shard (built once up front, held in RAM) -- not the first
few batches of the sorted val shards, which was one month's first ~70 games and
differed from run to run. With v2 shards the headline metric (best-checkpoint,
LR scheduler, early stopping) is top-1/top-3 over LEGAL moves only, since that
is what the engine actually plays; the unmasked numbers are printed alongside.

Example (toy CPU smoke test):
    python src/train.py --train-dir data/toy/train --val-dir data/toy/val \\
        --out-dir checkpoints/toy --batch-size 32 --d-model 64 --nhead 4 \\
        --num-layers 2 --dim-feedforward 128 --val-interval 50 --max-steps 500

Pass --drive-remote gdrive:chess_bot/checkpoints/run1/ to rclone-sync every
new-best checkpoint off the instance as it's saved (requires rclone installed
and configured -- see docs/reference/Teknoloji_Yigini_ve_Kaynaklar.md).

Pass --resume-from <checkpoint>.pt to continue training if a run was cut
short (pod died, watchdog fired, etc.): restores model weights, optimizer
state, the LR scheduler's own plateau-tracking state, and the
step/best-val_top1 counters from that checkpoint. The scheduler restore
matters: a freshly-constructed ReduceLROnPlateau starts its internal "best"
at -inf, so without this a resumed run can mistake several below-best val
checks for improvements and delay a real LR decay until early stopping
fires first (real incident, run6, 2026-09-26 -- see docs/log/Ilerleme_Notlari.md).

Regularization: AdamW (--weight-decay, default 0.01) instead of plain Adam,
and label smoothing on both heads' cross-entropy (--label-smoothing, default
0.1, 0 disables it). Dropout is a model hyperparameter (--dropout, already
existed). Added after run2 (4 months, ~9M positions) plateaued at val_top1
45.35% after ~2.1 epochs -- see docs/log/Ilerleme_Notlari.md.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import ChessTransformer, unpack_bits

MODEL_ARG_NAMES = ("d_model", "nhead", "num_layers", "dim_feedforward", "dropout", "gab_per_layer")
# A best.pt is ~60MB. 180s (real incident, run6, 2026-09-25 fix) assumed
# nothing slower than ~2.7Mbps and turned out too tight: a later run
# (2026-09-28) hit a pod with a genuinely slow but working connection
# (~250KB/s), needing ~244s -- the sync kept timing out and several real
# new-best checkpoints (including the run's true best) never made it to
# Drive before the pod was destroyed. 600s comfortably covers a legitimately
# slow transfer (>=100KB/s) while still catching an actual hang well before
# anyone would think to check on it.
RCLONE_SYNC_TIMEOUT_SECONDS = 600


class ShardDataset(Dataset):
    """Indexes shard_*.npz files from one or more directories as one flat
    dataset (e.g. combining several months' worth of build_dataset.py runs --
    each run numbers its own shards from 0, so collisions only matter within
    a single directory, not across separate ones).

    Keeps only the current shard's arrays in memory, so this scales past
    what fits in RAM as long as one shard does. Pair with ShardShuffledSampler
    (not plain shuffle=True) or every __getitem__ can reload a different
    shard from disk -- see ShardShuffledSampler's docstring.
    """

    def __init__(self, shard_dirs: Path | list[Path]):
        if isinstance(shard_dirs, (str, Path)):
            shard_dirs = [shard_dirs]
        self.files = sorted(f for d in shard_dirs for f in Path(d).glob("shard_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no shard_*.npz files in {shard_dirs}")
        lengths = []
        self.has_extras = None  # shard format v2 (legal + attacks arrays); v1 shards lack them
        for f in self.files:
            with np.load(f) as d:
                lengths.append(len(d["moves"]))
                has_extras = "legal" in d.files and "attacks" in d.files
            if self.has_extras is None:
                self.has_extras = has_extras
            elif has_extras != self.has_extras:
                raise ValueError(f"{f} mixes shard formats with the other shards (v1 without legal/attacks, v2 with)")
        self._offsets = np.cumsum([0] + lengths)
        self._cache_idx = None
        self._cache_boards = None
        self._cache_moves = None
        self._cache_legal = None
        self._cache_attacks = None

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def _load_shard(self, shard_idx: int) -> None:
        if self._cache_idx != shard_idx:
            with np.load(self.files[shard_idx]) as d:
                self._cache_boards = d["boards"]
                self._cache_moves = d["moves"]
                if self.has_extras:
                    self._cache_legal = d["legal"]
                    self._cache_attacks = d["attacks"]
            self._cache_idx = shard_idx

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        shard_idx = int(np.searchsorted(self._offsets, idx, side="right") - 1)
        self._load_shard(shard_idx)
        local_idx = idx - self._offsets[shard_idx]
        board = self._cache_boards[local_idx].astype(np.float32)
        move = self._cache_moves[local_idx].astype(np.int64)
        return torch.from_numpy(board), torch.from_numpy(move)

    def __getitems__(self, indices: list[int]) -> tuple:
        """Whole-batch fetch (DataLoader calls this instead of one __getitem__
        per sample when it exists -- pair with collate_fn=_identity): numpy
        fancy-indexes each involved shard's cached arrays once per batch
        instead of paying Python per-sample overhead 256 times, which was the
        real bottleneck (~6.5 steps/s regardless of GPU).

        Returns (boards uint8 (B,C,8,8), moves int64 (B,2), legal uint8
        (B,512) | None, attacks uint8 (B,64,8) | None), in `indices` order;
        boards stay uint8 (the model casts on-device) so host-to-device
        copies are 4x smaller."""
        idx = np.asarray(indices, dtype=np.int64)
        shard_ids = np.searchsorted(self._offsets, idx, side="right") - 1
        positions, boards, moves, legal, attacks = [], [], [], [], []
        for shard_idx in np.unique(shard_ids):
            sel = np.nonzero(shard_ids == shard_idx)[0]
            self._load_shard(int(shard_idx))
            local = idx[sel] - self._offsets[shard_idx]
            positions.append(sel)
            boards.append(self._cache_boards[local])
            moves.append(self._cache_moves[local])
            if self.has_extras:
                legal.append(self._cache_legal[local])
                attacks.append(self._cache_attacks[local])
        restore = np.argsort(np.concatenate(positions))  # back to the caller's order

        def gather(parts):
            return torch.from_numpy(np.concatenate(parts)[restore])

        return (
            gather(boards),
            gather(moves).long(),
            gather(legal) if self.has_extras else None,
            gather(attacks) if self.has_extras else None,
        )


def _identity(batch):
    return batch  # __getitems__ already returns a finished batch


class ShardShuffledSampler(Sampler):
    """Shuffles shard order each epoch, and shuffles within each shard, but
    never interleaves shards -- consecutive yielded indices stay in the same
    shard until it's exhausted, so ShardDataset's single-shard cache stays
    warm instead of reloading a different file from disk on every item.
    Plain DataLoader(shuffle=True) does global random access and defeats
    the cache -- fine at toy scale, but disk-bound (and GPU-idle, which you
    pay for) at real scale.
    """

    def __init__(self, dataset: ShardDataset):
        self.dataset = dataset

    def __iter__(self):
        rng = np.random.default_rng()
        for shard_idx in rng.permutation(len(self.dataset.files)):
            start, end = self.dataset._offsets[shard_idx], self.dataset._offsets[shard_idx + 1]
            for local_idx in rng.permutation(end - start):
                yield int(start + local_idx)

    def __len__(self) -> int:
        return len(self.dataset)


def build_eval_set(dataset: ShardDataset, num_positions: int, chunk: int = 4096) -> tuple:
    """A fixed, evenly-strided sample of `num_positions` positions across the
    WHOLE dataset (every shard, hence every month and many games), held in RAM
    as (boards, moves, legal|None, attacks|None) -- same batch layout as
    ShardDataset.__getitems__. Strided indices are visited in ascending order
    so each shard is read from disk once."""
    total = len(dataset)
    idx = np.unique(np.linspace(0, total - 1, min(num_positions, total)).astype(np.int64))
    parts = [dataset.__getitems__(idx[i:i + chunk].tolist()) for i in range(0, len(idx), chunk)]
    return tuple(None if parts[0][k] is None else torch.cat([p[k] for p in parts]) for k in range(4))


def _topk_hits(scores: torch.Tensor, true_idx: torch.Tensor) -> tuple[int, int]:
    top1 = (scores.argmax(dim=1) == true_idx).sum().item()
    top3 = (scores.topk(3, dim=1).indices == true_idx.unsqueeze(1)).any(dim=1).sum().item()
    return top1, top3


@torch.no_grad()
def evaluate(model: nn.Module, eval_set: tuple, device: torch.device, batch_size: int = 512) -> dict[str, float]:
    """top1/top3 over all 4096 from-to pairs (the pre-run9 metric), plus
    legal_top1/legal_top3 over legal pairs only when the eval set has legal
    masks (shard v2)."""
    boards_all, moves_all, legal_all, _ = eval_set
    model.eval()
    n = len(moves_all)
    hits = dict(top1=0, top3=0, legal_top1=0, legal_top3=0)
    for start in range(0, n, batch_size):
        boards = boards_all[start:start + batch_size].to(device)
        moves = moves_all[start:start + batch_size].to(device)
        joint = model.joint_logits(boards).reshape(boards.shape[0], -1)
        true_idx = moves[:, 0] * 64 + moves[:, 1]
        hits["top1"], hits["top3"] = (a + b for a, b in zip((hits["top1"], hits["top3"]), _topk_hits(joint, true_idx)))
        if legal_all is not None:
            legal = unpack_bits(legal_all[start:start + batch_size].to(device))
            masked = joint.masked_fill(~legal, float("-inf"))
            hits["legal_top1"], hits["legal_top3"] = (
                a + b for a, b in zip((hits["legal_top1"], hits["legal_top3"]), _topk_hits(masked, true_idx))
            )
    model.train()
    result = {k: v / n for k, v in hits.items()}
    if legal_all is None:
        del result["legal_top1"], result["legal_top3"]
    return result


def headline(metrics: dict[str, float]) -> tuple[float, float]:
    """(top1, top3) that drive best-checkpoint/LR-scheduler/early-stopping:
    legal-masked when available (what the engine plays), else unmasked."""
    if "legal_top1" in metrics:
        return metrics["legal_top1"], metrics["legal_top3"]
    return metrics["top1"], metrics["top3"]


def format_metrics(metrics: dict[str, float]) -> str:
    top1, top3 = headline(metrics)
    text = f"val_top1 {top1:.4f} val_top3 {top3:.4f}"
    if "legal_top1" in metrics:
        text += f" (unmasked {metrics['top1']:.4f}/{metrics['top3']:.4f})"
    return text


def save_checkpoint(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau, args: argparse.Namespace,
    step: int, top1: float, top3: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "model_args": {name: getattr(args, name) for name in MODEL_ARG_NAMES},
            "step": step,
            "val_top1": top1,
            "val_top3": top3,
        },
        path,
    )
    if args.drive_remote:
        # Sync every new-best checkpoint off the instance as it happens, so
        # training survives an SSH drop / local machine going to sleep
        # without losing progress if the instance itself dies. A sync
        # failure (missing rclone binary, transient network blip, bad
        # remote) must not kill training -- log it and move on, the next
        # new-best checkpoint will retry. Real incident (run6, 2026-09-25):
        # rclone hung indefinitely mid-upload (root cause unconfirmed --
        # possibly a stalled OAuth token refresh right after a fresh
        # `rclone config`) with no timeout on this call, silently blocking
        # the entire training loop for 10+ minutes until manually killed.
        try:
            result = subprocess.run(
                ["rclone", "copy", str(path), args.drive_remote], capture_output=True, text=True,
                timeout=RCLONE_SYNC_TIMEOUT_SECONDS,
            )
            if result.returncode != 0:
                print(f"  WARNING: rclone sync failed ({result.returncode}): {result.stderr.strip()[:300]}", file=sys.stderr)
            else:
                print(f"  synced to {args.drive_remote}", file=sys.stderr)
        except OSError as e:
            print(f"  WARNING: rclone sync failed to start: {e}", file=sys.stderr)
        except subprocess.TimeoutExpired:
            print(f"  WARNING: rclone sync timed out after {RCLONE_SYNC_TIMEOUT_SECONDS}s -- continuing without backup", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dir", required=True, type=Path, nargs="+", help="One or more directories of train shards (e.g. one per month)")
    parser.add_argument("--val-dir", required=True, type=Path, nargs="+", help="One or more directories of val shards")
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/run"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many steps regardless of epoch")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Cross-entropy label smoothing (0 disables it)")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-interval", type=int, default=5000, help="Evaluate val every N steps")
    parser.add_argument(
        "--patience", type=int, default=None,
        help="Stop early after this many consecutive val checks with no top-1 improvement (unset = disabled, run the full --epochs)",
    )
    parser.add_argument(
        "--val-positions", type=int, default=51_200,
        help="Size of the fixed, evenly-strided val sample drawn from every val shard (built once, evaluated each --val-interval)",
    )
    parser.add_argument("--log-interval", type=int, default=50, help="Log train loss every N steps")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--gab-per-layer", action="store_true",
        help="Give every encoder layer its own GAB bias instead of one shared by all layers (run9)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--drive-remote", default=None,
        help="If set (e.g. gdrive:chess_bot/checkpoints/run1/), rclone-copy every new-best checkpoint here as it's saved",
    )
    parser.add_argument(
        "--resume-from", type=Path, default=None,
        help="Resume from a checkpoint saved by this script: restores model weights, optimizer state, "
             "LR scheduler state, step count and best val_top1 (optimizer/scheduler state are skipped, "
             "with a warning, if the checkpoint predates them -- the scheduler falls back to seeding "
             "just its \"best\" with the true best_top1 instead of leaving it at -inf). Training then "
             "continues from that step count; the epoch loop still restarts at epoch 0 since shards are "
             "reshuffled each run anyway.",
    )
    parser.add_argument(
        "--lr-patience", type=int, default=1,
        help="Halve (see --lr-factor) the learning rate after this many consecutive val checks with no "
             "top-1 improvement. Should be < --patience so a decayed LR gets a chance before early stopping.",
    )
    parser.add_argument("--lr-factor", type=float, default=0.5, help="Multiply the learning rate by this on each --lr-patience plateau")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    train_ds = ShardDataset(args.train_dir)
    val_ds = ShardDataset(args.val_dir)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=ShardShuffledSampler(train_ds),
        num_workers=args.num_workers, collate_fn=_identity,
    )
    eval_set = build_eval_set(val_ds, args.val_positions)
    print(
        f"train positions: {len(train_ds)}, val positions: {len(val_ds)} "
        f"(fixed eval sample: {len(eval_set[1])}, legal masks: {eval_set[2] is not None})",
        file=sys.stderr,
    )

    model = ChessTransformer(
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout, gab_per_layer=args.gab_per_layer,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=args.lr_factor, patience=args.lr_patience)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    writer = SummaryWriter(log_dir=str(args.out_dir / "tb"))
    ckpt_path = args.out_dir / "best.pt"

    best_top1 = -1.0
    step = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        else:
            print(f"  WARNING: {args.resume_from} has no optimizer state (older checkpoint) -- optimizer starts fresh", file=sys.stderr)
        step = ckpt["step"]
        best_top1 = ckpt["val_top1"]
        # ReduceLROnPlateau tracks its own "best" internally, separate from
        # best_top1 above -- a freshly-constructed scheduler starts that at
        # -inf, so right after resuming it treats the next several val
        # checks as "improving" relative to -inf even when they're actually
        # below the true best_top1, delaying a real decay. Real incident
        # (run6, 2026-09-26): this let early stopping (which correctly
        # compares against best_top1) fire before the scheduler ever got a
        # second chance to decay after a forced resume. Restore its full
        # state when the checkpoint has one (saved by this same fix); older
        # checkpoints don't, so fall back to seeding just `.best` with the
        # true value instead of leaving it at -inf.
        if "scheduler_state_dict" in ckpt:
            lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            lr_scheduler.best = best_top1
            print(f"  WARNING: {args.resume_from} has no scheduler state (older checkpoint) -- seeding scheduler.best={best_top1:.4f} instead of -inf", file=sys.stderr)
        print(f"resumed from {args.resume_from} at step {step}, best val_top1 {best_top1:.4f}", file=sys.stderr)

    checks_without_improvement = 0
    start = time.time()
    stop = False
    for epoch in range(args.epochs):
        if stop:
            break
        epoch_bar = tqdm(train_loader, desc=f"epoch {epoch}", unit="step", dynamic_ncols=True)
        for boards, moves, _legal, _attacks in epoch_bar:
            boards, moves = boards.to(device), moves.to(device)
            from_logits, to_logits = model(boards)
            loss = criterion(from_logits, moves[:, 0]) + criterion(to_logits, moves[:, 1])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            epoch_bar.set_postfix(loss=f"{loss.item():.4f}", step=step, best_top1=f"{best_top1:.4f}")

            if step % args.log_interval == 0:
                tqdm.write(f"step {step} epoch {epoch} loss {loss.item():.4f} ({time.time() - start:.0f}s)")
                writer.add_scalar("train/loss", loss.item(), step)

            if step % args.val_interval == 0:
                metrics = evaluate(model, eval_set, device)
                top1, top3 = headline(metrics)
                lr_before = optimizer.param_groups[0]["lr"]
                lr_scheduler.step(top1)
                lr_after = optimizer.param_groups[0]["lr"]
                writer.add_scalar("val/top1", top1, step)
                writer.add_scalar("val/top3", top3, step)
                writer.add_scalar("train/lr", lr_after, step)
                tqdm.write(f"step {step} {format_metrics(metrics)} lr {lr_after:.2e}")
                if lr_after != lr_before:
                    tqdm.write(f"  lr decayed {lr_before:.2e} -> {lr_after:.2e} (plateaued {args.lr_patience} checks)")
                if top1 > best_top1:
                    best_top1 = top1
                    checks_without_improvement = 0
                    save_checkpoint(ckpt_path, model, optimizer, lr_scheduler, args, step, top1, top3)
                    tqdm.write(f"  new best checkpoint -> {ckpt_path}")
                else:
                    checks_without_improvement += 1
                    if args.patience is not None and checks_without_improvement >= args.patience:
                        tqdm.write(f"  early stopping: no val_top1 improvement in {checks_without_improvement} checks")
                        stop = True
                        break

            if args.max_steps is not None and step >= args.max_steps:
                stop = True
                break
        epoch_bar.close()

    if best_top1 < 0:  # never hit a val-interval boundary (e.g. max-steps < val-interval)
        metrics = evaluate(model, eval_set, device)
        top1, top3 = headline(metrics)
        best_top1 = top1
        save_checkpoint(ckpt_path, model, optimizer, lr_scheduler, args, step, top1, top3)
        print(f"final {format_metrics(metrics)}, checkpoint -> {ckpt_path}", file=sys.stderr)

    writer.close()
    print(f"done: {step} steps, best val_top1 {best_top1:.4f}, checkpoint at {ckpt_path}")


if __name__ == "__main__":
    main()
