#!/usr/bin/env bash
set -uo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root" || exit 1

echo "===== H20 FULLMASK GENERATION READINESS AUDIT ====="
date
hostname
whoami
pwd

echo
echo "===== git ====="
git log -1 --oneline || true
git status --short || true

echo
echo "===== gpu ====="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits || nvidia-smi || true

echo
echo "===== disk ====="
df -h /home/nvme01 /home/nvme02 /home/nvme03 /home/nvme04 "$repo_root" 2>/dev/null || df -h "$repo_root" || true

first_existing() {
  for path in "$@"; do
    if [ -n "$path" ] && [ -e "$path" ]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  return 1
}

echo
echo "===== candidate paths ====="
train_yaml="$(first_existing \
  "${VIDEO_DPO_TRAIN_DATA_YAML:-}" \
  /home/nvme01/H20_Video_inpainting_DPO/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.absolute.yaml \
  /home/nvme01/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.pai.yaml \
  /home/nvme01/VideoDPO/configs/vc2_dpo/vidpro/train_data.yaml \
  /home/nvme01/H20_Video_inpainting_DPO/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.pai.yaml \
  /home/nvme01/H20_Video_inpainting_DPO/external/VideoDPO/configs/vc2_dpo/vidpro/train_data.yaml \
  /mnt/nas/hj/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.pai.yaml || true)"
diffueraser_py="$(first_existing \
  "${DIFFUERASER_PYTHON:-}" \
  /home/nvme01/conda_envs/diffueraser/bin/python \
  "$(dirname "$repo_root")/conda_envs/diffueraser/bin/python" \
  /mnt/nas/hj/conda_envs/diffueraser/bin/python || true)"
audit_python="$(first_existing \
  "${PYTHON:-}" \
  "${VIDEODPO_PYTHON:-}" \
  "${PROPAINTER_PYTHON:-}" \
  /home/nvme01/conda_envs/videodpo/bin/python \
  /mnt/nas/hj/conda_envs/videodpo/bin/python \
  "$(command -v python 2>/dev/null || true)" \
  "$(command -v python3 2>/dev/null || true)" || true)"
third_party_root="$(first_existing \
  "${THIRD_PARTY_VIDEO_INPAINTING_ROOT:-}" \
  /home/nvme01/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting \
  "$repo_root/third_party_video_inpainting" \
  /mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting || true)"
base_model="$(first_existing \
  "${BASE_MODEL_PATH:-}" \
  "$third_party_root/downloads/sd_inpaint_hf_extract/stable-diffusion-inpainting" \
  /home/nvme01/weights/stable-diffusion-inpainting \
  /mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/downloads/sd_inpaint_hf_extract/stable-diffusion-inpainting || true)"
vae_path="$(first_existing \
  "${VAE_PATH:-}" \
  /home/nvme01/weights/sd-vae-ft-mse \
  "$repo_root/weights/sd-vae-ft-mse" \
  /mnt/nas/hj/weights/sd-vae-ft-mse || true)"
diffueraser_weights="$(first_existing \
  "${DIFFUERASER_WEIGHT_ROOT:-}" \
  "$third_party_root/weights/diffueraser/Orign_Diffueraser" \
  "$third_party_root/weights/diffuEraser" \
  "$repo_root/weights/diffueraser" \
  /mnt/nas/hj/H20_Video_inpainting_DPO_scp_backup_20260515_101902/third_party_video_inpainting/weights/diffueraser/Orign_Diffueraser || true)"
propainter_weights="$(first_existing \
  "${PROPAINTER_WEIGHT_ROOT:-}" \
  "$third_party_root/weights/propainter" \
  /home/nvme01/data/third_party_video_inpainting/weights/propainter \
  /mnt/nas/hj/data/third_party_video_inpainting/weights/propainter || true)"
pcm_weights="$(first_existing \
  "${PCM_WEIGHTS_PATH:-}" \
  /home/nvme01/weights/PCM_Weights \
  "$repo_root/weights/PCM_Weights" \
  "$third_party_root/weights/PCM_Weights" \
  "$third_party_root/downloads/PCM_Weights" \
  /mnt/nas/hj/weights/PCM_Weights || true)"

printf 'VIDEO_DPO_TRAIN_DATA_YAML=%s\n' "${train_yaml:-MISSING}"
printf 'ORCHESTRATION_PYTHON=%s\n' "${audit_python:-MISSING}"
printf 'DIFFUERASER_PYTHON=%s\n' "${diffueraser_py:-MISSING}"
printf 'THIRD_PARTY_VIDEO_INPAINTING_ROOT=%s\n' "${third_party_root:-MISSING}"
printf 'BASE_MODEL_PATH=%s\n' "${base_model:-MISSING}"
printf 'VAE_PATH=%s\n' "${vae_path:-MISSING}"
printf 'DIFFUERASER_WEIGHT_ROOT=%s\n' "${diffueraser_weights:-MISSING}"
printf 'PROPAINTER_WEIGHT_ROOT=%s\n' "${propainter_weights:-MISSING}"
printf 'PCM_WEIGHTS_PATH=%s\n' "${pcm_weights:-MISSING}"

echo
echo "===== python/env smoke ====="
if [ -n "$diffueraser_py" ]; then
  "$diffueraser_py" - <<'PY' || true
import sys
print("python", sys.executable)
print("version", sys.version.split()[0])
for name in ["torch", "cv2", "numpy", "PIL", "diffusers"]:
    try:
        mod = __import__(name)
        print("import", name, "OK", getattr(mod, "__version__", ""))
    except Exception as exc:
        print("import", name, "FAILED", repr(exc))
PY
else
  echo "missing DIFFUERASER_PYTHON; skip import smoke"
fi

echo
echo "===== canonical VideoDPO setting ====="
if [ -n "$train_yaml" ] && [ -n "$audit_python" ]; then
  VIDEO_DPO_TRAIN_DATA_YAML="$train_yaml" "$audit_python" - <<'PY' || true
from pathlib import Path
from tools.pai_videodpo_single_sample_generation_smoke import load_yaml, resolve_videodpo_roots, read_json

train_yaml = Path(__import__("os").environ["VIDEO_DPO_TRAIN_DATA_YAML"]).resolve()
roots = resolve_videodpo_roots(train_yaml)
root = roots[0]
pairs = read_json(root / "pair.json")
print("train_yaml", train_yaml)
print("resolved_data_root", root)
print("pair_count", len(pairs))
cfg = load_yaml(Path("DPO_finetune/configs/official_diffueraser_stage1.yaml"))
params = cfg["data"]["params"]["train"]["params"]
print("canonical_num_frames", int(params.get("video_length", 16)))
print("canonical_height", int(params.get("train_height") or params.get("resolution", [320,512])[0]))
print("canonical_width", int(params.get("train_width") or params.get("resolution", [320,512])[-1]))
PY
else
  echo "missing train yaml or orchestration python; cannot resolve canonical data"
fi

echo
echo "===== audit decision hints ====="
echo "- Required for D1 small sample: train yaml, diffueraser python, base model, VAE, DiffuEraser weights, ProPainter weights for DiffuEraser wrapper, PCM weights, free disk, idle GPU."
echo "- Do not start full generation until LIMIT=20 and LIMIT=100 pass manifest/video checks."
