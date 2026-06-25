#!/usr/bin/env bash
set -euo pipefail

TARGET_USER=hj
TARGET_GROUP="$(id -gn "$TARGET_USER")"
BASE=/mnt/nas/hj/H20_Video_inpainting_DPO

need_user() {
  id "$TARGET_USER" >/dev/null
}

need_user

if ! command -v setfacl >/dev/null 2>&1; then
  cat >&2 <<'MSG'
ERROR: setfacl is not installed on this PAI image.
Install/enable ACL tooling first. Do not chmod 777 shared weights.
For Exp-only output directories, precise chown is used below, but shared weights require setfacl.
MSG
  exit 2
fi

# 1) Shared DiffuEraser weights: grant hj read/traverse only on the required asset subtree.
# Do not change owner. Do not grant write. Do not touch the whole weights root recursively.
for d in \
  /mnt/nas/hj \
  /mnt/nas/hj/weights \
  /mnt/nas/hj/weights/diffuEraser; do
  [ -e "$d" ] && setfacl -m u:${TARGET_USER}:--x "$d"
done
setfacl -R -m u:${TARGET_USER}:rX /mnt/nas/hj/weights/diffuEraser/converted_weights_step48000

# 2) Exp-only output dirs: create/own only these experiment-specific directories.
for d in \
  "$BASE/logs/autoresearch/exp25_vor_or_preference_data" \
  "$BASE/logs/autoresearch/exp26_videopainter_dpo_v2" \
  "$BASE/logs/autoresearch/exp27_paper_grounded_preference_study" \
  "$BASE/experiments/dpo/exp25_vor_or_preference_data" \
  "$BASE/experiments/dpo/exp26_videopainter_dpo_v2" \
  "$BASE/experiments/dpo/exp27_paper_grounded_preference_study" \
  "$BASE/runtime"; do
  install -d -o "$TARGET_USER" -g "$TARGET_GROUP" -m 2770 "$d"
  setfacl -m u:${TARGET_USER}:rwx "$d"
  setfacl -m d:u:${TARGET_USER}:rwx "$d"
done

# 3) Verify as hj.
runuser -u "$TARGET_USER" -- test -x /mnt/nas/hj/weights/diffuEraser/converted_weights_step48000
runuser -u "$TARGET_USER" -- test -r /mnt/nas/hj/weights/diffuEraser/converted_weights_step48000
runuser -u "$TARGET_USER" -- bash -lc 'find /mnt/nas/hj/weights/diffuEraser/converted_weights_step48000 -maxdepth 3 -type f \( -name config.json -o -name "*.safetensors" -o -name "*.bin" \) | head -5 >/tmp/diffueraser_weight_read_probe.txt && test -s /tmp/diffueraser_weight_read_probe.txt'
for d in \
  "$BASE/logs/autoresearch/exp25_vor_or_preference_data" \
  "$BASE/logs/autoresearch/exp26_videopainter_dpo_v2" \
  "$BASE/logs/autoresearch/exp27_paper_grounded_preference_study" \
  "$BASE/experiments/dpo/exp25_vor_or_preference_data" \
  "$BASE/experiments/dpo/exp26_videopainter_dpo_v2" \
  "$BASE/experiments/dpo/exp27_paper_grounded_preference_study" \
  "$BASE/runtime"; do
  runuser -u "$TARGET_USER" -- test -w "$d"
done

echo PAI_POSTMAINTENANCE_MINIMAL_PERMISSIONS_OK
