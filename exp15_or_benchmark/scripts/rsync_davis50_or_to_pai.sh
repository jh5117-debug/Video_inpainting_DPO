#!/usr/bin/env bash
set -euo pipefail

HAL_DAVIS_ROOT="${HAL_DAVIS_ROOT:-/home/hj/Video_inpainting_DPO/data/external/davis_2017_full_resolution/DAVIS}"
PAI_HOST="${PAI_HOST:-root@47.103.26.60}"
PAI_PORT="${PAI_PORT:-22}"
PAI_TARGET="${PAI_TARGET:-/mnt/nas/hj/data/external/davis_2017_full_resolution_or_eval50/DAVIS}"
MANIFEST_DIR="${MANIFEST_DIR:-exp15_or_benchmark/manifests}"
FILES_FROM="${FILES_FROM:-${MANIFEST_DIR}/davis50_or_rsync_files.txt}"

cd "$(dirname "$0")/../.."
test -d "$HAL_DAVIS_ROOT"
test -f "$FILES_FROM"

echo "===== rsync DAVIS50 OR subset to PAI ====="
date
echo "HAL_DAVIS_ROOT=$HAL_DAVIS_ROOT"
echo "PAI_HOST=$PAI_HOST"
echo "PAI_TARGET=$PAI_TARGET"

ssh -o StrictHostKeyChecking=no -p "$PAI_PORT" "$PAI_HOST" "mkdir -p '$PAI_TARGET'"

rsync -avh --partial --append-verify \
  -e "ssh -o StrictHostKeyChecking=no -p ${PAI_PORT}" \
  --files-from "$FILES_FROM" \
  "$HAL_DAVIS_ROOT"/ \
  "$PAI_HOST:$PAI_TARGET"/

ssh -o StrictHostKeyChecking=no -p "$PAI_PORT" "$PAI_HOST" "\
  set -e; \
  ws_root=/mnt/workspace/hj/nas_hj/data/external/davis_2017_full_resolution_or_eval50; \
  mkdir -p /mnt/workspace/hj/nas_hj/data/external; \
  if [ -L \"\$ws_root\" ]; then \
    :; \
  elif [ -e \"\$ws_root/DAVIS\" ]; then \
    :; \
  elif [ -e \"\$ws_root\" ]; then \
    echo \"workspace target exists but is not usable: \$ws_root\" >&2; \
    exit 2; \
  else \
    ln -s '$PAI_TARGET' \"\$ws_root\"; \
  fi; \
  du -sh '$PAI_TARGET'; \
  find '$PAI_TARGET/JPEGImages/Full-Resolution' -mindepth 1 -maxdepth 1 -type d | wc -l; \
  find '$PAI_TARGET/Annotations/Full-Resolution' -mindepth 1 -maxdepth 1 -type d | wc -l"

echo "===== done ====="
