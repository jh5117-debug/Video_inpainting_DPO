#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root"

stamp="$(date +%Y%m%d_%H%M%S)"
report="${REPORT:-$repo_root/PRD/pai_audit_pai_node_${stamp}.md}"
readiness_report="${READINESS_REPORT:-$repo_root/PRD/pai_asset_readiness_report.md}"
env_file="${ENV_FILE:-$repo_root/configs/paths/pai.detected.env}"

mkdir -p "$(dirname "$report")" "$(dirname "$env_file")"

ROOTS=(
  "$repo_root"
  "/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO"
  "/mnt/nas/hj/H20_Video_inpainting_DPO"
  "/mnt/nas/hj"
  "/mnt/workspace/hj"
  "/home/hj"
)

write_line() {
  printf '%s\n' "$*" >> "$report"
}

section() {
  write_line
  write_line "## $*"
}

run_block() {
  local title="$1"
  shift
  section "$title"
  write_line '```text'
  "$@" >> "$report" 2>&1 || true
  write_line '```'
}

first_existing() {
  local p
  for p in "$@"; do
    if [ -n "$p" ] && [ -e "$p" ]; then
      printf '%s\n' "$p"
      return 0
    fi
  done
  return 1
}

dirname_if_file() {
  local p="$1"
  if [ -n "$p" ] && [ -f "$p" ]; then
    dirname "$p"
  fi
}

find_first_dir() {
  local pattern="$1"
  local root
  local found
  for root in "${ROOTS[@]}"; do
    [ -d "$root" ] || continue
    found="$(timeout 60s find "$root" -maxdepth 7 -type d -iname "$pattern" 2>/dev/null | head -1 || true)"
    if [ -n "$found" ]; then
      printf '%s\n' "$found"
      return 0
    fi
  done
}

find_first_file() {
  local pattern="$1"
  local root
  local found
  for root in "${ROOTS[@]}"; do
    [ -d "$root" ] || continue
    found="$(timeout 60s find "$root" -maxdepth 8 -type f -iname "$pattern" 2>/dev/null | head -1 || true)"
    if [ -n "$found" ]; then
      printf '%s\n' "$found"
      return 0
    fi
  done
}

link_current() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"
  if [ -z "$src" ] || [ ! -e "$src" ]; then
    write_line "- MISSING: $dst"
    return 0
  fi
  rm -rf "$dst"
  ln -s "$src" "$dst"
  write_line "- LINKED: $dst -> $src"
}

emit_env() {
  local key="$1"
  local value="$2"
  if [ -n "$value" ]; then
    printf 'export %s=%q\n' "$key" "$value" >> "$env_file"
  else
    printf '# export %s=UNCONFIRMED\n' "$key" >> "$env_file"
  fi
}

echo "# PAI Audit And Asset Preparation" > "$report"
write_line
write_line "- generated_at: $(date -Is)"
write_line "- repo_root: $repo_root"
write_line "- env_file: $env_file"
write_line "- readiness_report: $readiness_report"

run_block "Basic Info" bash -lc 'date; hostname; whoami; pwd'
run_block "Git Info" bash -lc 'git rev-parse --show-toplevel 2>/dev/null; git branch --show-current 2>/dev/null || true; git status --short; git log -1 --oneline || true'
run_block "GPU Info" bash -lc 'nvidia-smi || true'
run_block "Python Info" bash -lc 'which python || true; python --version || true; which pip || true; pip --version || true; conda info --envs || true'
run_block "Important Python Packages" python - <<'PY'
import importlib
mods = [
    "torch", "torchvision", "diffusers", "transformers", "accelerate",
    "decord", "cv2", "imageio", "moviepy", "av", "numpy", "PIL",
    "wandb", "einops", "omegaconf",
]
for name in mods:
    try:
        mod = importlib.import_module(name)
        print(f"{name}: OK {getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{name}: MISSING or ERROR: {exc}")
PY

section "Known Completed Experiment Outputs"
write_line '```text'
for p in \
  "$repo_root/logs/videodpo_vc2_dpo_official_clean/pai-vc2-dpo-official-full-gpu0-3-gb8-step3000-20260521_061414" \
  "$repo_root/logs/vbench_full/pai-vc2-official-step3000-full-vbench-20260521_141824" \
  "$repo_root/logs/official_diffueraser_stage1/pai-official-diffueraser-stage1-full-gpu4-7-gb8-step3000-20260521_072559" \
  "$repo_root/logs/official_diffueraser_stage2/pai-official-diffueraser-stage2-full-gpu4-7-gb8-step3000-20260521_150540" \
  "$repo_root/logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926" \
  "$repo_root/logs/qual_sbs_30/vc2_and_diffueraser_20260522"; do
  if [ -e "$p" ]; then
    printf 'FOUND %s\n' "$p"
  else
    printf 'MISSING %s\n' "$p"
  fi
done >> "$report"
write_line '```'

official_videodpo_root="$(first_existing \
  "${VIDEO_DPO_OFFICIAL_ROOT:-}" \
  "/mnt/nas/hj/official_repos/VideoDPO_official_1febdb4" \
  "$repo_root/external/VideoDPO_official_1febdb4" \
  "$repo_root/official_repos/VideoDPO_official_1febdb4" \
  "$(find_first_dir '*VideoDPO_official*')" || true)"

videodpo_data_root="$(first_existing \
  "${VIDEO_DPO_DATA_ROOT:-}" \
  "$repo_root/data/videodpo/current" \
  "$official_videodpo_root/data" \
  "$official_videodpo_root/datasets" \
  "$repo_root/data/external/videodpo" \
  "$(find_first_dir '*videodpo*data*')" || true)"

youtube_vos_root="$(first_existing \
  "${YOUTUBE_VOS_ROOT:-}" \
  "$repo_root/data/youtubevos/current" \
  "$repo_root/data/external/youtubevos_432_240" \
  "$repo_root/data/external/ytbv_2019_full_resolution" \
  "/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/youtubevos_432_240" \
  "/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/ytbv_2019_full_resolution" \
  "$(find_first_dir '*youtubevos*')" \
  "$(find_first_dir '*ytbv*')" || true)"

generated_loser_root="$(first_existing \
  "${GENERATED_LOSER_ROOT:-}" \
  "$repo_root/data/generated_losers" || true)"

diffueraser_weight_root="$(first_existing \
  "${DIFFUERASER_WEIGHT_ROOT:-}" \
  "$repo_root/weights/diffueraser/current" \
  "$repo_root/weights/diffuEraser/converted_weights_step48000" \
  "$repo_root/weights/diffuEraser" \
  "/mnt/nas/hj/H20_Video_inpainting_DPO/weights/diffuEraser/converted_weights_step48000" \
  "/mnt/nas/hj/H20_Video_inpainting_DPO/weights/diffuEraser" \
  "/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/diffueraser/Orign_Diffueraser" \
  "/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/diffueraser" \
  "$(find_first_dir '*diffueraser*weight*')" || true)"

propainter_weight_root="$(first_existing \
  "${PROPAINTER_WEIGHT_ROOT:-}" \
  "$repo_root/weights/propainter/current" \
  "$(dirname_if_file "$repo_root/weights/propainter/ProPainter.pth")" \
  "$repo_root/weights/ProPainter" \
  "/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter" \
  "$(dirname_if_file "/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter/ProPainter.pth")" \
  "/mnt/nas/hj/H20_Video_inpainting_DPO/weights/ProPainter" \
  "$(dirname_if_file "/mnt/nas/hj/H20_Video_inpainting_DPO/weights/propainter/ProPainter.pth")" \
  "/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/propainter" \
  "$(dirname_if_file "/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/propainter/ProPainter.pth")" \
  "$(find_first_dir '*propainter*')" || true)"

cococo_weight_root="$(first_existing \
  "${COCOCO_WEIGHT_ROOT:-}" \
  "$repo_root/weights/cococo/current" \
  "/mnt/nas/hj/H20_Video_inpainting_DPO/weights/COCOCO_weight" \
  "/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/COCOCO_weight" \
  "/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/cococo" \
  "$(find_first_dir '*cococo*')" || true)"

minimax_weight_root="$(first_existing \
  "${MINIMAX_REMOVER_WEIGHT_ROOT:-}" \
  "$repo_root/weights/minimax_remover/current" \
  "/mnt/nas/hj/H20_Video_inpainting_DPO/weights/minimax_remover" \
  "/mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/minimax" \
  "$HOME/.cache/huggingface/hub/models--zibojia--minimax-remover" \
  "/root/.cache/huggingface/hub/models--zibojia--minimax-remover" \
  "$(find_first_dir '*minimax*')" || true)"

official_videodpo_weight_root="$(first_existing \
  "${OFFICIAL_VIDEODPO_WEIGHT_ROOT:-}" \
  "$official_videodpo_root/checkpoints" \
  "$official_videodpo_root" || true)"

vc2_weight_root="$(first_existing \
  "${VC2_WEIGHT_ROOT:-}" \
  "$official_videodpo_root/checkpoints/vc2" \
  "$repo_root/weights/vc2/current" \
  "$(find_first_file 'model.ckpt' | xargs -r dirname)" || true)"

section "Detected Data And Weight Roots"
write_line '```text'
for kv in \
  "VIDEO_DPO_OFFICIAL_ROOT=$official_videodpo_root" \
  "VIDEO_DPO_DATA_ROOT=$videodpo_data_root" \
  "YOUTUBE_VOS_ROOT=$youtube_vos_root" \
  "GENERATED_LOSER_ROOT=$generated_loser_root" \
  "DIFFUERASER_WEIGHT_ROOT=$diffueraser_weight_root" \
  "PROPAINTER_WEIGHT_ROOT=$propainter_weight_root" \
  "COCOCO_WEIGHT_ROOT=$cococo_weight_root" \
  "MINIMAX_REMOVER_WEIGHT_ROOT=$minimax_weight_root" \
  "OFFICIAL_VIDEODPO_WEIGHT_ROOT=$official_videodpo_weight_root" \
  "VC2_WEIGHT_ROOT=$vc2_weight_root"; do
  printf '%s\n' "$kv"
done >> "$report"
write_line '```'

: > "$env_file"
emit_env VIDEO_DPO_OFFICIAL_ROOT "$official_videodpo_root"
emit_env VIDEO_DPO_DATA_ROOT "$videodpo_data_root"
emit_env YOUTUBE_VOS_ROOT "$youtube_vos_root"
emit_env GENERATED_LOSER_ROOT "$generated_loser_root"
emit_env DIFFUERASER_WEIGHT_ROOT "$diffueraser_weight_root"
emit_env PROPAINTER_WEIGHT_ROOT "$propainter_weight_root"
emit_env COCOCO_WEIGHT_ROOT "$cococo_weight_root"
emit_env MINIMAX_REMOVER_WEIGHT_ROOT "$minimax_weight_root"
emit_env OFFICIAL_VIDEODPO_WEIGHT_ROOT "$official_videodpo_weight_root"
emit_env VC2_WEIGHT_ROOT "$vc2_weight_root"
emit_env EXP_OUTPUT_ROOT "${EXP_OUTPUT_ROOT:-$repo_root/outputs}"

section "Prepared Current Symlinks"
link_current "$videodpo_data_root" "$repo_root/data/videodpo/current"
link_current "$youtube_vos_root" "$repo_root/data/youtubevos/current"
link_current "$generated_loser_root" "$repo_root/data/generated_losers/current"
link_current "$diffueraser_weight_root" "$repo_root/weights/diffueraser/current"
link_current "$propainter_weight_root" "$repo_root/weights/propainter/current"
link_current "$cococo_weight_root" "$repo_root/weights/cococo/current"
link_current "$minimax_weight_root" "$repo_root/weights/minimax_remover/current"
link_current "$official_videodpo_weight_root" "$repo_root/weights/official_videodpo/current"
link_current "$vc2_weight_root" "$repo_root/weights/vc2/current"

run_block "Four Inpainting Model Search" bash -lc 'find . /home/hj /mnt/workspace /mnt/data /mnt/nas/hj -maxdepth 6 \( -iname "*diffueraser*" -o -iname "*propainter*" -o -iname "*cococo*" -o -iname "*minimax*" -o -iname "*remover*" \) 2>/dev/null | sort | head -500'
run_block "Generation Script Search" bash -lc 'find . -type f \( -name "*.py" -o -name "*.sh" -o -name "*.sbatch" \) | grep -Ei "generate|infer|inpaint|propainter|cococo|minimax|diffueraser|remover" | sort | head -300'
run_block "Large Asset Check" bash -lc 'find data weights outputs experiments -maxdepth 4 -type f -size +50M -print 2>/dev/null | sort || true'
run_block "Disk Capacity" bash -lc 'df -h; du -sh data weights outputs logs 2>/dev/null || true; du -sh /mnt/nas/hj/data /mnt/workspace/hj 2>/dev/null || true'

section "Next Step"
write_line "- Source detected env on PAI:"
write_line
write_line '```bash'
write_line "source \"$env_file\""
write_line '```'
write_line
write_line "- If any root above is blank or marked MISSING in the symlink section, that dataset/weight is still unconfirmed and should be downloaded or pointed to before launching the corresponding experiment."

echo "[pai-audit] report=$report"
cp "$report" "$readiness_report"
echo "[pai-audit] readiness_report=$readiness_report"
echo "[pai-audit] env_file=$env_file"
