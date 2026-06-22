#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
for f in exp25_vor_or_preference_data/runtime/transfer.pid exp25_vor_or_preference_data/runtime/monitor.pid; do
  [ -f "$f" ] || continue
  pid="$(cat "$f")"
  if [ -n "$pid" ] && [ -d "/proc/$pid" ]; then
    cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" || true)"
    case "$cmd" in
      *exp25_vor_or_preference_data*|*effecterase*)
        kill "$pid" || true
        ;;
      *)
        echo "Refusing to stop unrelated pid $pid: $cmd" >&2
        ;;
    esac
  fi
done

