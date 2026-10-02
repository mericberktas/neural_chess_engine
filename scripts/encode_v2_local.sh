#!/usr/bin/env bash
# Encode the harvested filtered PGNs to shard format v2 (boards + moves + legal
# masks + attack bitboards) under data/v2/, for the run9 a/b/c comparison:
# the 16 train months run6 used plus the held-out test month. Local CPU only,
# no GPU. Resumable: a month that finished gets a .done marker and is skipped
# on re-run; a month that was interrupted halfway is wiped and redone.
#
# Runs from a SNAPSHOT of src/ taken at start, not the live checkout -- real
# incident (2026-10-02): switching git branches while this ran made a month's
# Python start hit "PermissionError: Access denied" on a src file git was
# rewriting at that moment.
#
#   bash scripts/encode_v2_local.sh
#   then: rclone copy data/v2 gdrive:chess_bot/tensors/v2/ --transfers 8
set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/Scripts/python.exe
WORKERS="${WORKERS:-14}"
TRAIN_MONTHS="2026-08 2026-07 2026-06 2026-05 2026-04 2026-03 2026-02 2026-01 2025-12 2025-11 2025-10 2025-09 2025-08 2025-07 2025-06 2025-05"
TEST_MONTH="2025-04"

SNAP="$(mktemp -d)"
cp -r src/. "$SNAP/"
echo "encoding from snapshot $SNAP"

encode() {  # month out_dir split max_games
    local month="$1" out="$2" split="$3" max_games="$4"
    if [ -f "$out/.done" ]; then echo "== $month: already done, skipping"; return; fi
    rm -rf "$out"
    echo "== $month ($split)"
    "$PY" -u "$SNAP/build_dataset.py" --source "data/filtered_pgn/$month.pgn" --out-dir "$out" \
        --split "$split" --max-games "$max_games" --workers "$WORKERS" 2>&1 | tail -n 1
    # build_dataset ends with os._exit(0), so check the output rather than the exit code
    if ls "$out"/*/shard_00000.npz > /dev/null 2>&1; then touch "$out/.done"; else echo "!! $month produced no shards"; fi
}

for month in $TRAIN_MONTHS; do encode "$month" "data/v2/$month" train 40000; done
encode "$TEST_MONTH" "data/v2/test_$TEST_MONTH" test 5000
echo ALL_DONE
