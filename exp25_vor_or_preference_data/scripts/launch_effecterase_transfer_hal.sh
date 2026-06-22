#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p exp25_vor_or_preference_data/runtime exp25_vor_or_preference_data/logs reports experiment_registry/exp25_vor_or_preference_data

LOCK="exp25_vor_or_preference_data/runtime/transfer.lock"
PID_FILE="exp25_vor_or_preference_data/runtime/transfer.pid"
PGID_FILE="exp25_vor_or_preference_data/runtime/transfer.pgid"
MON_PID_FILE="exp25_vor_or_preference_data/runtime/monitor.pid"
LOG="exp25_vor_or_preference_data/logs/effecterase_hal_to_pai_transfer.log"
MON_LOG="exp25_vor_or_preference_data/logs/effecterase_transfer_monitor.outer.log"
PY="/home/hj/.venvs/hf_effecterase/bin/python"

if [ -f "$PID_FILE" ]; then
  pid="$(cat "$PID_FILE")"
  if [ -n "$pid" ] && [ -d "/proc/$pid" ]; then
    cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" || true)"
    if [[ "$cmd" == *"transfer_effecterase_hal_to_pai.py"* ]]; then
      echo "Transfer already running pid=$pid"
      exit 0
    fi
  fi
fi

flock -n "$LOCK" bash -c '
  nohup setsid nice -n 15 ionice -c2 -n7 "$0" exp25_vor_or_preference_data/scripts/transfer_effecterase_hal_to_pai.py \
    > "$1" 2>&1 < /dev/null &
  pid=$!
  echo "$pid" > "$2"
  sleep 1
  pgid="$(ps -o pgid= -p "$pid" | tr -d " " || true)"
  echo "$pgid" > "$3"
' "$PY" "$LOG" "$PID_FILE" "$PGID_FILE"

if [ -f "$MON_PID_FILE" ]; then
  mpid="$(cat "$MON_PID_FILE")"
  if [ -n "$mpid" ] && [ -d "/proc/$mpid" ]; then
    mcmd="$(tr '\0' ' ' < "/proc/$mpid/cmdline" || true)"
    if [[ "$mcmd" == *"monitor_effecterase_transfer.py"* ]]; then
      echo "Monitor already running pid=$mpid"
      exit 0
    fi
  fi
fi

nohup setsid "$PY" exp25_vor_or_preference_data/scripts/monitor_effecterase_transfer.py \
  > "$MON_LOG" 2>&1 < /dev/null &
echo "$!" > "$MON_PID_FILE"

echo "transfer_pid=$(cat "$PID_FILE")"
echo "transfer_pgid=$(cat "$PGID_FILE")"
echo "monitor_pid=$(cat "$MON_PID_FILE")"
echo "log=$LOG"
echo "monitor_log=$MON_LOG"

