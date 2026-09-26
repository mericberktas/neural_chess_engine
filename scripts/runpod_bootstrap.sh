#!/usr/bin/env bash
# Bootstrap a rented RunPod instance for a real training run.
# Run this ON the instance after SSH-ing in (not on your local machine).
#
# Before running this script:
#   1. From your LOCAL machine: create the pod from a PyTorch image
#      (torch/CUDA already installed there -- this script deliberately does
#      not reinstall torch).
#   2. SSH in, then set up Drive access ON THE INSTANCE directly:
#        rclone authorize drive
#      This prints a URL and waits; open an SSH session from your local
#      machine with `-L 53682:localhost:53682` to that same instance, open
#      the printed URL in your own browser to complete Google's OAuth
#      consent, then paste the resulting token into an rclone.conf on the
#      instance (see docs/reference/Teknoloji_Yigini_ve_Kaynaklar.md for the
#      full walkthrough). Never scp an rclone.conf from your local machine
#      to the instance -- that file holds a live Drive OAuth token, and
#      moving it off your machine is a real credential transfer (blocked in
#      practice, 2026-09-25, when tried that way).
#   3. Then run this script.
#
# Configurable via env vars (all optional, shown with defaults):
#   REPO_URL, TRAIN_MONTH, TEST_MONTH, WORKDIR, TRAIN_MAX_GAMES, TEST_MAX_GAMES
#
# TRAIN_MAX_GAMES/TEST_MAX_GAMES are load-bearing, not cosmetic: without a
# cap, build_dataset.py streams the ENTIRE monthly dump (tens of millions of
# games) since there's no way to stop early once you're past the "few
# million positions" target -- observed live at ~500 games/sec, ~3% qualify,
# that's many hours to days on a meter that's still running. The plan's own
# target is "a few million positions", not "the whole month".
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/mericberktas/neural_chess_engine.git}"
TRAIN_MONTH="${TRAIN_MONTH:-2026-08}"
TEST_MONTH="${TEST_MONTH:-2026-04}"
WORKDIR="${WORKDIR:-/workspace/neural_chess_engine}"
TRAIN_MAX_GAMES="${TRAIN_MAX_GAMES:-40000}"   # ~2.3M positions at ~57 positions/game observed
TEST_MAX_GAMES="${TEST_MAX_GAMES:-5000}"      # held-out eval set doesn't need to be huge

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
    --out-dir data/train_full --split train --max-games "$TRAIN_MAX_GAMES"

echo "== held-out test month (time-disjoint from training, per Aşama 1) =="
python src/build_dataset.py \
    --source "https://database.lichess.org/standard/lichess_db_standard_rated_${TEST_MONTH}.pgn.zst" \
    --out-dir data/test_full --split test --max-games "$TEST_MAX_GAMES"

echo ""
echo "== setup done. Starting training in the background (survives SSH drop / your machine sleeping) =="
echo "== every new-best checkpoint auto-syncs to gdrive:chess_bot/checkpoints/run1/ as it's saved =="
nohup python src/train.py \
    --train-dir data/train_full/train --val-dir data/train_full/val \
    --out-dir checkpoints/run1 \
    --batch-size 256 --d-model 256 --nhead 8 --num-layers 6 --dim-feedforward 1024 \
    --val-interval 2000 --epochs 1 \
    --drive-remote gdrive:chess_bot/checkpoints/run1/ \
    > train.log 2>&1 &
echo "training pid: $!  (tail -f $WORKDIR/train.log to watch progress)"
echo ""
echo "== when training is done (check train.log for 'done:'), DESTROY the pod from your LOCAL machine =="
echo "==   (stopping alone keeps billing storage -- destroy is the only way to stop all charges) =="
