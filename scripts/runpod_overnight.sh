#!/usr/bin/env bash
# Unattended overnight run: several months of train data + a held-out test
# month, early-stopped training, then the pod terminates ITSELF -- for when
# nobody is around to watch it or manually destroy it afterward.
#
# Needs /root/.runpod_key on the instance FIRST: a RESTRICTED (Pods read/write
# only, no billing/account access) RunPod API key -- never the account's main
# key. This script uses it only to terminate this one pod when done or if the
# watchdog fires. Get one at runpod.io/console/user/settings -> API Keys.
#
# Transmitting that key from a Windows/PowerShell client: `$key | ssh host
# "cat > /root/.runpod_key"` adds a UTF-8 BOM (﻿) at the start of the
# piped text. It's invisible and .strip() does NOT remove it (BOM isn't
# whitespace) -- it silently broke terminate_pod()'s HTTP header encoding on
# 2026-09-22/23 and left a pod running for ~6 extra billed hours after
# training had already finished successfully. Reading the key file with
# `encoding='utf-8-sig'` (done below) strips a leading BOM if present and is
# a no-op if it's not -- safe either way, keep it even if the transmission
# method changes.
#
# Run this ON the instance after SSH-ing in (not on your local machine):
#   bash scripts/runpod_overnight.sh
#
# Configurable via env vars (all optional, shown with defaults):
#   REPO_URL, WORKDIR, TRAIN_MONTHS (space-separated), TEST_MONTH,
#   TRAIN_MAX_GAMES, TEST_MAX_GAMES, EPOCHS, PATIENCE, WATCHDOG_HOURS
set -uo pipefail
# deliberately NOT set -e: if a step fails partway through the night, we still
# want execution to reach the final self-terminate call, not hang forever
# billing on a stalled/crashed script. The watchdog below is the second,
# independent layer of that same guarantee.

REPO_URL="${REPO_URL:-https://github.com/mericberktas/neural_chess_engine.git}"
WORKDIR="${WORKDIR:-/workspace/neural_chess_engine}"
TRAIN_MONTHS="${TRAIN_MONTHS:-2026-08 2026-07 2026-06 2026-05}"
TEST_MONTH="${TEST_MONTH:-2026-04}"
TRAIN_MAX_GAMES="${TRAIN_MAX_GAMES:-40000}"
TEST_MAX_GAMES="${TEST_MAX_GAMES:-5000}"
EPOCHS="${EPOCHS:-15}"
PATIENCE="${PATIENCE:-3}"
WATCHDOG_HOURS="${WATCHDOG_HOURS:-8}"
KEY_FILE=/root/.runpod_key

terminate_self() {
    if [ -f "$KEY_FILE" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
        python -c "
import runpod
runpod.api_key = open('$KEY_FILE', encoding='utf-8-sig').read().strip()
runpod.terminate_pod('$RUNPOD_POD_ID')
print('self-terminated pod $RUNPOD_POD_ID')
" 2>&1
    else
        echo "!! cannot self-terminate: missing $KEY_FILE or \$RUNPOD_POD_ID (pod id was: '${RUNPOD_POD_ID:-}')" >&2
    fi
}

echo "RUNPOD_POD_ID=${RUNPOD_POD_ID:-<not set>}"

# Hard safety net: whatever happens above, kill this pod after WATCHDOG_HOURS
# regardless -- covers a hang or crash that never reaches the normal
# self-terminate call at the end of this script. Dies along with the pod
# once that happens, so nothing to clean up if the normal path wins the race.
( sleep "$((WATCHDOG_HOURS * 3600))"; echo "WATCHDOG FIRED after ${WATCHDOG_HOURS}h -- self-terminating"; terminate_self ) &
WATCHDOG_PID=$!

if [ -d "$WORKDIR/.git" ]; then
    (cd "$WORKDIR" && git pull)
else
    git clone "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR" || { echo "!! cd $WORKDIR failed"; terminate_self; exit 1; }

echo "== python deps (skipping torch -- already in the image) =="
grep -v '^torch$' requirements.txt > /tmp/requirements-no-torch.txt
pip install --quiet -r /tmp/requirements-no-torch.txt
pip install --quiet runpod

echo "== rclone =="
if ! command -v rclone &> /dev/null; then
    curl -s https://rclone.org/install.sh | bash
fi
if [ ! -f /root/.config/rclone/rclone.conf ]; then
    echo "!! /root/.config/rclone/rclone.conf not found -- checkpoints won't be backed up."
fi
rclone lsd gdrive: > /dev/null 2>&1 && echo "rclone OK, gdrive: remote reachable"

TRAIN_DIRS=()
VAL_DIRS=()
for month in $TRAIN_MONTHS; do
    echo "== train data: $month =="
    python src/build_dataset.py \
        --source "https://database.lichess.org/standard/lichess_db_standard_rated_${month}.pgn.zst" \
        --out-dir "data/train_${month}" --split train --max-games "$TRAIN_MAX_GAMES"
    TRAIN_DIRS+=("data/train_${month}/train")
    VAL_DIRS+=("data/train_${month}/val")
done

echo "== held-out test month: $TEST_MONTH =="
python src/build_dataset.py \
    --source "https://database.lichess.org/standard/lichess_db_standard_rated_${TEST_MONTH}.pgn.zst" \
    --out-dir data/test_full --split test --max-games "$TEST_MAX_GAMES"

echo "== training: cap ${EPOCHS} epochs, early stop after ${PATIENCE} non-improving val checks =="
python src/train.py \
    --train-dir "${TRAIN_DIRS[@]}" --val-dir "${VAL_DIRS[@]}" \
    --out-dir checkpoints/run2 \
    --batch-size 256 --d-model 256 --nhead 8 --num-layers 6 --dim-feedforward 1024 \
    --val-interval 2000 --epochs "$EPOCHS" --patience "$PATIENCE" \
    --drive-remote gdrive:chess_bot/checkpoints/run2/

echo "== training finished, self-terminating =="
kill "$WATCHDOG_PID" 2>/dev/null
terminate_self
