"""Train the ChessTransformer policy network on build_dataset.py shards.

Reads train/ and val/ subfolders of shard_*.npz files (boards: (N,18,8,8)
uint8, moves: (N,2) int16 = from_square, to_square). Val is evaluated every
--val-interval steps (not once per epoch, since one epoch over the real
dataset can be huge) and the checkpoint with the best val top-1 is kept.

Example (toy CPU smoke test):
    python src/train.py --train-dir data/toy/train --val-dir data/toy/val \\
        --out-dir checkpoints/toy --batch-size 32 --d-model 64 --nhead 4 \\
        --num-layers 2 --dim-feedforward 128 --val-interval 50 --max-steps 500
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter

from model import ChessTransformer

MODEL_ARG_NAMES = ("d_model", "nhead", "num_layers", "dim_feedforward", "dropout")


class ShardDataset(Dataset):
    """Indexes a directory of shard_*.npz files as one flat dataset.

    Keeps only the current shard's arrays in memory, so this scales past
    what fits in RAM as long as one shard does. Pair with ShardShuffledSampler
    (not plain shuffle=True) or every __getitem__ can reload a different
    shard from disk -- see ShardShuffledSampler's docstring.
    """

    def __init__(self, shard_dir: Path):
        self.files = sorted(Path(shard_dir).glob("shard_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no shard_*.npz files in {shard_dir}")
        lengths = []
        for f in self.files:
            with np.load(f) as d:
                lengths.append(len(d["moves"]))
        self._offsets = np.cumsum([0] + lengths)
        self._cache_idx = None
        self._cache_boards = None
        self._cache_moves = None

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def _load_shard(self, shard_idx: int) -> None:
        if self._cache_idx != shard_idx:
            with np.load(self.files[shard_idx]) as d:
                self._cache_boards = d["boards"]
                self._cache_moves = d["moves"]
            self._cache_idx = shard_idx

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        shard_idx = int(np.searchsorted(self._offsets, idx, side="right") - 1)
        self._load_shard(shard_idx)
        local_idx = idx - self._offsets[shard_idx]
        board = self._cache_boards[local_idx].astype(np.float32)
        move = self._cache_moves[local_idx].astype(np.int64)
        return torch.from_numpy(board), torch.from_numpy(move)


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


@torch.no_grad()
def evaluate(model: nn.Module, val_loader: DataLoader, device: torch.device, max_batches: int | None) -> tuple[float, float]:
    model.eval()
    top1_correct = top3_correct = n = 0
    for i, (boards, moves) in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        boards, moves = boards.to(device), moves.to(device)
        from_logits, to_logits = model(boards)
        joint = (from_logits.unsqueeze(-1) + to_logits.unsqueeze(-2)).reshape(boards.shape[0], -1)
        true_idx = moves[:, 0] * 64 + moves[:, 1]
        top1_correct += (joint.argmax(dim=1) == true_idx).sum().item()
        top3_idx = joint.topk(3, dim=1).indices
        top3_correct += (top3_idx == true_idx.unsqueeze(1)).any(dim=1).sum().item()
        n += boards.shape[0]
    model.train()
    return top1_correct / n, top3_correct / n


def save_checkpoint(path: Path, model: nn.Module, args: argparse.Namespace, step: int, top1: float, top3: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_args": {name: getattr(args, name) for name in MODEL_ARG_NAMES},
            "step": step,
            "val_top1": top1,
            "val_top3": top3,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dir", required=True, type=Path)
    parser.add_argument("--val-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/run"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many steps regardless of epoch")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-interval", type=int, default=5000, help="Evaluate val every N steps")
    parser.add_argument("--val-batches", type=int, default=20, help="Batches per val evaluation (caps eval time)")
    parser.add_argument("--log-interval", type=int, default=50, help="Log train loss every N steps")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    train_ds = ShardDataset(args.train_dir)
    val_ds = ShardDataset(args.val_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=ShardShuffledSampler(train_ds), num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"train positions: {len(train_ds)}, val positions: {len(val_ds)}", file=sys.stderr)

    model = ChessTransformer(
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=str(args.out_dir / "tb"))
    ckpt_path = args.out_dir / "best.pt"

    best_top1 = -1.0
    step = 0
    start = time.time()
    stop = False
    for epoch in range(args.epochs):
        if stop:
            break
        for boards, moves in train_loader:
            boards, moves = boards.to(device), moves.to(device)
            from_logits, to_logits = model(boards)
            loss = criterion(from_logits, moves[:, 0]) + criterion(to_logits, moves[:, 1])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % args.log_interval == 0:
                print(f"step {step} epoch {epoch} loss {loss.item():.4f} ({time.time() - start:.0f}s)", file=sys.stderr)
                writer.add_scalar("train/loss", loss.item(), step)

            if step % args.val_interval == 0:
                top1, top3 = evaluate(model, val_loader, device, args.val_batches)
                writer.add_scalar("val/top1", top1, step)
                writer.add_scalar("val/top3", top3, step)
                print(f"step {step} val_top1 {top1:.4f} val_top3 {top3:.4f}", file=sys.stderr)
                if top1 > best_top1:
                    best_top1 = top1
                    save_checkpoint(ckpt_path, model, args, step, top1, top3)
                    print(f"  new best checkpoint -> {ckpt_path}", file=sys.stderr)

            if args.max_steps is not None and step >= args.max_steps:
                stop = True
                break

    if best_top1 < 0:  # never hit a val-interval boundary (e.g. max-steps < val-interval)
        top1, top3 = evaluate(model, val_loader, device, args.val_batches)
        best_top1 = top1
        save_checkpoint(ckpt_path, model, args, step, top1, top3)
        print(f"final val_top1 {top1:.4f} val_top3 {top3:.4f}, checkpoint -> {ckpt_path}", file=sys.stderr)

    writer.close()
    print(f"done: {step} steps, best val_top1 {best_top1:.4f}, checkpoint at {ckpt_path}")


if __name__ == "__main__":
    main()
