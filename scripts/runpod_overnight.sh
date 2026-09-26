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
# Also wants /root/.config/rclone/rclone.conf for Drive access (harvested
# PGN input + checkpoint backups) -- set this up ON THE INSTANCE directly via
# `rclone authorize drive` (needs a port-forwarded SSH session so you can
# complete the Google OAuth consent in your own local browser -- see
# docs/reference/Teknoloji_Yigini_ve_Kaynaklar.md). Never scp an rclone.conf
# from your local machine to the instance -- that file holds a live Drive
# OAuth token and moving it off your machine is treated as a real credential
# transfer, blocked in practice (2026-09-25) when tried this way.
#
# To restart train.py mid-run WITHOUT losing the pod (e.g. to deploy a code
# fix): `touch /root/PAUSE_NO_TERMINATE` BEFORE stopping it. Without this,
# ANY exit of train.py -- crash, natural early-stop, or a manual kill -- is
# treated as "finished" and self-terminates the pod (see the `set -uo
# pipefail` note below for why).
#
# Run this ON the instance after SSH-ing in (not on your local machine):
#   bash scripts/runpod_overnight.sh
#
# Configurable via env vars (all optional, shown with defaults):
#   REPO_URL, WORKDIR, GIT_BRANCH (default main -- e.g. set to
#   run5-rich-encoding to run an experiment branch without a manual
#   `git checkout` step), RUN_NAME, TRAIN_MONTHS (space-separated), TEST_MONTH,
#   TRAIN_MAX_GAMES, TEST_MAX_GAMES, EPOCHS, PATIENCE, WATCHDOG_HOURS,
#   RESUME_FROM_REMOTE (an rclone path to a checkpoint, e.g.
#   gdrive:chess_bot/checkpoints/run3/best.pt -- fetched and passed to
#   train.py's --resume-from if set), FILTERED_PGN_REMOTE (default
#   gdrive:chess_bot/filtered_pgn/ -- see scripts/harvest_filtered_pgn.py),
#   NUM_MONTHS (if set, overrides TRAIN_MONTHS/TEST_MONTH: picks the N most
#   recent months available in FILTERED_PGN_REMOTE as train, the next-oldest
#   one as test), TENSOR_REMOTE (default gdrive:chess_bot/tensors/ -- caches
#   encoded shards per RUN_NAME/month so a re-run of the same run skips the
#   CPU-bound build_dataset.py step entirely on a cache hit)
#
# Per-month data comes from FILTERED_PGN_REMOTE when a month's file is
# there (fast: a few-MB rclone copy, no re-scanning Lichess) and falls back
# to streaming straight from database.lichess.org otherwise (slow, but
# works for a month nobody has harvested yet). See
# scripts/harvest_filtered_pgn.py -- run it locally ahead of time (no GPU
# needed) to populate FILTERED_PGN_REMOTE and skip the slow path entirely.
set -uo pipefail
# deliberately NOT set -e: if a step fails partway through the night, we still
# want execution to reach the final self-terminate call, not hang forever
# billing on a stalled/crashed script. The watchdog below is the second,
# independent layer of that same guarantee.
#
# This also means ANY exit of the training command below -- a crash, a
# natural early-stop, OR a deliberate manual kill during a live intervention
# (e.g. to deploy a code fix) -- reaches the same self-terminate call and
# destroys the pod. Real incident (run6, 2026-09-25): stopping train.py by
# hand to redeploy a fix was silently treated as "training finished" and
# destroyed the pod mid-fix. To safely restart train.py WITHOUT losing the
# pod: `touch /root/PAUSE_NO_TERMINATE` BEFORE stopping it -- see the check
# right after training exits, below.

LOG_FILE=/root/overnight.log
exec > >(tee -a "$LOG_FILE") 2>&1
PAUSE_FILE=/root/PAUSE_NO_TERMINATE

REPO_URL="${REPO_URL:-https://github.com/mericberktas/neural_chess_engine.git}"
WORKDIR="${WORKDIR:-/workspace/neural_chess_engine}"
GIT_BRANCH="${GIT_BRANCH:-main}"
RUN_NAME="${RUN_NAME:-run4}"
TRAIN_MONTHS="${TRAIN_MONTHS:-2026-08 2026-07 2026-06 2026-05 2026-04 2026-03 2026-02 2026-01}"
TEST_MONTH="${TEST_MONTH:-2025-12}"
TRAIN_MAX_GAMES="${TRAIN_MAX_GAMES:-40000}"
TEST_MAX_GAMES="${TEST_MAX_GAMES:-5000}"
EPOCHS="${EPOCHS:-15}"
PATIENCE="${PATIENCE:-4}"
WATCHDOG_HOURS="${WATCHDOG_HOURS:-8}"
RESUME_FROM_REMOTE="${RESUME_FROM_REMOTE-gdrive:chess_bot/checkpoints/run3/best.pt}"
FILTERED_PGN_REMOTE="${FILTERED_PGN_REMOTE:-gdrive:chess_bot/filtered_pgn/}"
TENSOR_REMOTE="${TENSOR_REMOTE:-gdrive:chess_bot/tensors/}"
NUM_MONTHS="${NUM_MONTHS:-}"
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
    (cd "$WORKDIR" && git fetch origin && git checkout "$GIT_BRANCH" && git pull origin "$GIT_BRANCH")
else
    git clone --branch "$GIT_BRANCH" "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR" || { echo "!! cd $WORKDIR failed"; terminate_self; exit 1; }
echo "== on branch $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD) =="

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

if [ -n "$NUM_MONTHS" ]; then
    echo "== NUM_MONTHS=$NUM_MONTHS set: picking months from $FILTERED_PGN_REMOTE =="
    mapfile -t AVAILABLE < <(rclone lsf "$FILTERED_PGN_REMOTE" | sed -n 's/\.pgn$//p' | sort -r)
    if [ "${#AVAILABLE[@]}" -lt "$((NUM_MONTHS + 1))" ]; then
        echo "!! only ${#AVAILABLE[@]} months harvested in $FILTERED_PGN_REMOTE, need NUM_MONTHS+1=$((NUM_MONTHS + 1)) (train+test)"; terminate_self; exit 1
    fi
    TRAIN_MONTHS="${AVAILABLE[*]:0:$NUM_MONTHS}"
    TEST_MONTH="${AVAILABLE[$NUM_MONTHS]}"
    echo "TRAIN_MONTHS=$TRAIN_MONTHS"
    echo "TEST_MONTH=$TEST_MONTH"
fi

# Prefer an already-harvested filtered .pgn from Drive (fast: a few-MB copy,
# no re-scanning the full monthly dump) over streaming straight from
# Lichess (slow, but the only option for a month nobody has harvested yet).
fetch_month_source() {
    local month="$1"
    local local_path="filtered_pgn/${month}.pgn"
    mkdir -p filtered_pgn
    if rclone copyto "${FILTERED_PGN_REMOTE}${month}.pgn" "$local_path" 2>/dev/null && [ -s "$local_path" ]; then
        echo "$local_path"
    else
        echo "https://database.lichess.org/standard/lichess_db_standard_rated_${month}.pgn.zst"
    fi
}

# True if `rclone lsf $1` lists at least one file -- used to check a tensor
# cache path for a hit without failing on a merely-empty/missing remote dir
# (rclone lsf on a nonexistent path just prints nothing, exit 0).
remote_has_files() {
    [ -n "$(rclone lsf "$1" 2>/dev/null)" ]
}

TRAIN_DIRS=()
VAL_DIRS=()
for month in $TRAIN_MONTHS; do
    TENSOR_PATH="${TENSOR_REMOTE}${RUN_NAME}/${month}/"
    if remote_has_files "$TENSOR_PATH"; then
        echo "== train data: $month (cached tensors from $TENSOR_PATH) =="
        rclone copy "$TENSOR_PATH" "data/train_${month}/"
    else
        SOURCE=$(fetch_month_source "$month")
        echo "== train data: $month (source: $SOURCE) =="
        python src/build_dataset.py \
            --source "$SOURCE" \
            --out-dir "data/train_${month}" --split train --max-games "$TRAIN_MAX_GAMES"
        timeout 300 rclone copy "data/train_${month}/" "$TENSOR_PATH" || echo "!! could not cache tensors for $month to $TENSOR_PATH"
    fi
    TRAIN_DIRS+=("data/train_${month}/train")
    VAL_DIRS+=("data/train_${month}/val")
done

TEST_TENSOR_PATH="${TENSOR_REMOTE}${RUN_NAME}/test_${TEST_MONTH}/"
if remote_has_files "$TEST_TENSOR_PATH"; then
    echo "== held-out test month: $TEST_MONTH (cached tensors from $TEST_TENSOR_PATH) =="
    rclone copy "$TEST_TENSOR_PATH" "data/test_full/"
else
    TEST_SOURCE=$(fetch_month_source "$TEST_MONTH")
    echo "== held-out test month: $TEST_MONTH (source: $TEST_SOURCE) =="
    # Observed twice (2026-09-23): when a harvested source file has far more
    # games than TEST_MAX_GAMES needs, build_dataset.py's multiprocessing.Pool
    # is left with a big prefetch backlog at the --max-games break, and Pool
    # cleanup on exit can hang forever -- a known class of Python
    # multiprocessing issue, not specific to this data. Doesn't happen for
    # train months since their harvested file size matches TRAIN_MAX_GAMES
    # closely. Fixed at the source in build_dataset.py itself (terminate()
    # + os._exit(), no join) -- timeout+pkill kept here as defense in depth
    # since that fix couldn't be verified against the exact Linux/fork
    # conditions that triggered this. test_full isn't needed for training
    # itself (val comes from each train month's own val/ split) so a
    # stuck/failed test build must not be allowed to hang an unattended run.
    if ! timeout 600 python src/build_dataset.py \
        --source "$TEST_SOURCE" \
        --out-dir data/test_full --split test --max-games "$TEST_MAX_GAMES"; then
        echo "!! test month build timed out or failed -- continuing without it (not needed for training)"
    fi
    pkill -9 -f "src/build_dataset.py --source $TEST_SOURCE" 2>/dev/null || true
    if [ -d data/test_full ]; then
        timeout 300 rclone copy data/test_full/ "$TEST_TENSOR_PATH" || echo "!! could not cache test tensors to $TEST_TENSOR_PATH"
    fi
fi

RESUME_ARGS=()
if [ -n "$RESUME_FROM_REMOTE" ]; then
    echo "== fetching resume checkpoint: $RESUME_FROM_REMOTE =="
    mkdir -p /root/resume
    if rclone copyto "$RESUME_FROM_REMOTE" /root/resume/resume.pt; then
        RESUME_ARGS=(--resume-from /root/resume/resume.pt)
    else
        echo "!! could not fetch $RESUME_FROM_REMOTE -- starting fresh instead"
    fi
fi

echo "== training: cap ${EPOCHS} epochs, early stop after ${PATIENCE} non-improving val checks =="
python src/train.py \
    --train-dir "${TRAIN_DIRS[@]}" --val-dir "${VAL_DIRS[@]}" \
    --out-dir "checkpoints/$RUN_NAME" \
    --batch-size 256 --d-model 256 --nhead 8 --num-layers 6 --dim-feedforward 1024 \
    --val-interval 2000 --epochs "$EPOCHS" --patience "$PATIENCE" \
    --drive-remote "gdrive:chess_bot/checkpoints/$RUN_NAME/" \
    "${RESUME_ARGS[@]}"
TRAIN_EXIT=$?
echo "== train.py exited with code $TRAIN_EXIT =="
kill "$WATCHDOG_PID" 2>/dev/null

# Best-effort: back up this run's own log to Drive no matter how/why train.py
# exited, since the pod (and this log) is gone for good once destroyed below.
# Real incident (run6, 2026-09-25): a pod died mid-run with no surviving
# record of why, making the actual stopping point/reason unrecoverable.
if [ -f /root/.config/rclone/rclone.conf ]; then
    timeout 60 rclone copy "$LOG_FILE" "gdrive:chess_bot/checkpoints/$RUN_NAME/" || echo "!! could not back up $LOG_FILE"
fi

if [ -f "$PAUSE_FILE" ]; then
    rm -f "$PAUSE_FILE"
    echo "== $PAUSE_FILE found -- NOT self-terminating, pod stays alive for a manual restart =="
else
    echo "== self-terminating (train.py exit code $TRAIN_EXIT) =="
    terminate_self
fi
