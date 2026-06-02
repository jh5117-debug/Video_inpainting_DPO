#!/usr/bin/env bash
set -euo pipefail

# Run on PAI.  This syncs D3 target-domain generated-loser data from H20 into
# the canonical generated_losers directory.  It never uses --delete.

H20_HOST="${H20_HOST:-ubuntu@27.190.15.128}"
H20_ROOT="${H20_ROOT:-/home/nvme01/H20_Video_inpainting_DPO}"
PAI_ROOT="${PAI_ROOT:-/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO}"
D3_REL="data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4"
SRC_ROOT="${H20_ROOT}/${D3_REL}"
DST_ROOT="${PAI_ROOT}/${D3_REL}"
SYNC_MODE="${SYNC_MODE:-slim}" # slim or full
SLIM_LIST="${SLIM_LIST:-${DST_ROOT}/reports/d3_slim_rsync_files.txt}"
LOG_ROOT="${PAI_ROOT}/logs/data_sync"
SSH_OPTS="${SSH_OPTS:--o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=20}"
RSYNC_OPTS="${RSYNC_OPTS:--aH --partial --append-verify --info=progress2}"
DRY_RUN="${DRY_RUN:-false}"

mkdir -p "${LOG_ROOT}" "${DST_ROOT}/reports"

log() {
  printf '[d3-sync %(%F %T)T] %s\n' -1 "$*"
}

precheck() {
  log "precheck h20=${H20_HOST}"
  ssh ${SSH_OPTS} "${H20_HOST}" "hostname; date; test -d '${SRC_ROOT}'; du -sh '${SRC_ROOT}'; find '${SRC_ROOT}/_shards' -maxdepth 1 -type d -name 'shard_*' | wc -l"
  mkdir -p "${DST_ROOT}"
}

build_slim_list() {
  log "building slim rsync list from selected primary manifests"
  mkdir -p "$(dirname "${SLIM_LIST}")"
  ssh ${SSH_OPTS} "${H20_HOST}" "cd '${SRC_ROOT}' && python3 -" > "${SLIM_LIST}.tmp" <<'PY'
from __future__ import annotations
import json
from pathlib import Path

root = Path(".").resolve()
manifests = [
    Path("manifests/selected_primary_comp.jsonl"),
    Path("manifests/selected_primary_nocomp.jsonl"),
    Path("manifests/selection_events.jsonl"),
]
paths = set()
for manifest in manifests:
    if manifest.exists():
        paths.add(str(manifest))

path_fields = [
    "win_video_path",
    "mask_path",
    "raw_loser_video_path",
    "comp_loser_video_path",
    "final_loser_video_path",
]
for manifest in manifests[:2]:
    if not manifest.exists():
        continue
    with manifest.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            for field in path_fields:
                value = str(row.get(field, "") or "")
                if not value:
                    continue
                p = Path(value)
                if p.is_absolute():
                    try:
                        rel = p.relative_to(root)
                    except ValueError:
                        continue
                    paths.add(str(rel))
                else:
                    paths.add(value)

for rel in sorted(paths):
    print(rel.rstrip("/") + ("/" if not Path(rel).suffix else ""))
PY
  mv "${SLIM_LIST}.tmp" "${SLIM_LIST}"
  log "slim list=${SLIM_LIST} entries=$(wc -l < "${SLIM_LIST}")"
}

sync_slim() {
  build_slim_list
  local dry=()
  if [[ "${DRY_RUN}" == "true" ]]; then
    dry=(--dry-run)
  fi
  log "sync slim ${SRC_ROOT} -> ${DST_ROOT}"
  rsync ${RSYNC_OPTS} "${dry[@]}" --files-from="${SLIM_LIST}" -e "ssh ${SSH_OPTS}" "${H20_HOST}:${SRC_ROOT}/" "${DST_ROOT}/"
}

sync_full() {
  local dry=()
  if [[ "${DRY_RUN}" == "true" ]]; then
    dry=(--dry-run)
  fi
  log "sync full ${SRC_ROOT} -> ${DST_ROOT}"
  rsync ${RSYNC_OPTS} "${dry[@]}" -e "ssh ${SSH_OPTS}" "${H20_HOST}:${SRC_ROOT}/" "${DST_ROOT}/"
}

postcheck() {
  log "postcheck"
  du -sh "${DST_ROOT}" || true
  find "${DST_ROOT}" -type f | wc -l || true
  find "${DST_ROOT}/manifests" -maxdepth 1 -type f -printf '%s %p\n' 2>/dev/null | sort -n || true
}

main() {
  cd "${PAI_ROOT}"
  precheck
  case "${SYNC_MODE}" in
    slim) sync_slim ;;
    full) sync_full ;;
    *) echo "[error] SYNC_MODE must be slim or full, got ${SYNC_MODE}" >&2; exit 2 ;;
  esac
  postcheck
  log "done"
}

main "$@"
