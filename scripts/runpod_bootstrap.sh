#!/usr/bin/env bash
# Bootstrap a rented RunPod instance for a real training run.
# Run this ON the instance after SSH-ing in (not on your local machine).
#
# Before running this script, from your LOCAL machine:
#   1. Create the pod from a PyTorch image (torch/CUDA already installed
#      there -- this script deliberately does not reinstall torch).
#   2. Copy your rclone config so checkpoints can be pulled off before the
#      instance is destroyed:
#        scp -P <SSH_PORT> ~/.config/rclone/rclone.conf root@<SSH_HOST>:/root/.config/rclone/rclone.conf
#   3. SSH in, then run this script.
#
# Configurable via env vars (all optional, shown with defaults):
#   REPO_URL, TRAIN_MONTH, TEST_MONTH, WORKDIR
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/mericberktas/neural_chess_engine.git}"
TRAIN_MONTH="${TRAIN_MONTH:-2026-08}"
TEST_MONTH="${TEST_MONTH:-2026-04}"
WORKDIR="${WORKDIR:-/workspace/neural_chess_engine}"

echo "== clone =="
git clone "$REPO_URL" "$WORKDIR"
cd "$WORKDIR"

echo "== python deps (skipping torch -- the image already has a CUDA-matched build) =="
grep -v '^torch$' requirements.txt > /tmp/requirements-no-torch.txt
pip install -r /tmp/requirements-no-torch.txt

echo "== rclone =="
if ! command -v rclone &> /dev/null; then
    curl https://rclone.org/install.sh | sudo bash
fi
if [ ! -f /root/.config/rclone/rclone.conf ]; then
    echo "!! /root/.config/rclone/rclone.conf not found."
    echo "!! scp it from your local machine (see this script's header), then re-run."
    exit 1
fi
rclone lsd gdrive: > /dev/null && echo "rclone OK, gdrive: remote reachable"

echo "== training data (downloaded here, not on your local machine -- see docs/plans/01_Veri_Toplama_ve_Hazirlama.md) =="
python src/build_dataset.py \
    --source "https://database.lichess.org/standard/lichess_db_standard_rated_${TRAIN_MONTH}.pgn.zst" \
    --out-dir data/train_full --split train

echo "== held-out test month (time-disjoint from training, per Aşama 1) =="
python src/build_dataset.py \
    --source "https://database.lichess.org/standard/lichess_db_standard_rated_${TEST_MONTH}.pgn.zst" \
    --out-dir data/test_full --split test

echo ""
echo "== setup done. Hyperparameters aren't baked in here (decide those first) -- example run: =="
cat <<'EOF'
python src/train.py \
    --train-dir data/train_full/train --val-dir data/train_full/val \
    --out-dir checkpoints/run1 \
    --batch-size 256 --d-model 256 --nhead 8 --num-layers 6 --dim-feedforward 1024 \
    --val-interval 2000 --epochs 1

# periodically (or after training finishes), pull the checkpoint off the instance:
rclone copy checkpoints/run1/best.pt gdrive:chess_bot/checkpoints/run1/

# when done, DESTROY the pod from your LOCAL machine (stopping alone keeps billing storage):
#   python -c "import runpod; runpod.api_key='...'; runpod.terminate_pod('POD_ID')"
#   (or via the RunPod console)
EOF
