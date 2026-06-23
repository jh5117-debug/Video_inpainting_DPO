#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp25_vor}"
ARCHIVE_DIR="${ARCHIVE_DIR:-/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/downloads/fa09dc61128ca0418a4a13364d97a08018ea9cc7}"
LOG_DIR="$ROOT/exp25_vor_or_preference_data/logs"
RUNTIME_DIR="$ROOT/exp25_vor_or_preference_data/runtime"
LOCK="$RUNTIME_DIR/vor_train_mask_index.lock"
PID_FILE="$RUNTIME_DIR/vor_train_mask_index.pid"
PGID_FILE="$RUNTIME_DIR/vor_train_mask_index.pgid"
LOG="$LOG_DIR/vor_train_mask_index.log"
COMMAND_FILE="$RUNTIME_DIR/vor_train_mask_index_command.sh"

mkdir -p "$LOG_DIR" "$RUNTIME_DIR" "$ROOT/reports"

cat > "$COMMAND_FILE" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
exec 9>"$LOCK"
flock -n 9 || {
  echo "VOR Train/MASK index already running under lock $LOCK" >&2
  exit 1
}
echo "\$\$" > "$PID_FILE"
ps -o pgid= "\$\$" | tr -d ' ' > "$PGID_FILE"
exec ionice -c2 -n7 nice -n 10 python exp25_vor_or_preference_data/scripts/index_vor_archive_members.py \
  --archive-dir "$ARCHIVE_DIR" \
  --groups VOR-Train VOR-Train-MASK \
  --output-csv reports/vor_train_mask_member_index.csv \
  --state-json exp25_vor_or_preference_data/runtime/vor_train_mask_member_index_state.json \
  --heartbeat-json exp25_vor_or_preference_data/runtime/vor_train_mask_member_index_heartbeat.json \
  --resume \
  --heartbeat-every 1000
EOF
chmod 755 "$COMMAND_FILE"

nohup setsid bash "$COMMAND_FILE" >> "$LOG" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
ps -o pgid= $! | tr -d ' ' > "$PGID_FILE"
echo "pid=$(cat "$PID_FILE")"
echo "pgid=$(cat "$PGID_FILE")"
echo "log=$LOG"
echo "heartbeat=$RUNTIME_DIR/vor_train_mask_member_index_heartbeat.json"
