"""Evaluate a trained checkpoint on shard directories with train.py's exact
metric (fixed evenly-strided sample, top-1/3 over all pairs and, with shard
v2, over legal moves only) -- so every arm/run is scored on the same positions
the same way, including pre-run9 checkpoints (run6, run8) that never saw it.

Example (val shards of the 16 train months, then the held-out test month):
    python scripts/eval_checkpoint.py --checkpoint checkpoints/run6/best.pt \\
        --shard-dir data/v2/2026-08/val data/v2/2026-07/val --positions 51200
    python scripts/eval_checkpoint.py --checkpoint checkpoints/run9a/best.pt \\
        --shard-dir data/v2/test_2025-04/test --positions 350000
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model import ChessTransformer  # noqa: E402
from train import ShardDataset, build_eval_set, evaluate, format_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--shard-dir", required=True, type=Path, nargs="+")
    parser.add_argument("--positions", type=int, default=51_200, help="Fixed strided sample size (cap; uses all if fewer exist)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ChessTransformer(**ckpt["model_args"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    dataset = ShardDataset(args.shard_dir)
    eval_set = build_eval_set(dataset, args.positions)
    metrics = evaluate(model, eval_set, device)
    print(f"{args.checkpoint} (step {ckpt['step']}): {len(eval_set[1])} positions, {format_metrics(metrics)}")


if __name__ == "__main__":
    main()
