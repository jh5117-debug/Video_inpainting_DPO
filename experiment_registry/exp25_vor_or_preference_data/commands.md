# Exp25 Commands

## CLI4 Gate16 DE-B

```bash
python -m py_compile exp25_vor_or_preference_data/scripts/select_gate16_deb_sources.py exp25_vor_or_preference_data/scripts/launch_exp25_gate16_deb.py
python -m unittest exp25_vor_or_preference_data.tests.test_gate16_deb_selection
python exp25_vor_or_preference_data/scripts/launch_exp25_gate16_deb.py --project-root /home/hj/runtime_code_snapshots/cli4_exp25_<commit> --compute-lpips --compute-ewarp
```

```bash
HF_HOME=/home/hj/.cache/huggingface_effecterase_auth /home/hj/.venvs/hf_effecterase/bin/hf auth whoami
/home/hj/.venvs/hf_effecterase/bin/python exp25_vor_or_preference_data/scripts/audit_hf_effecterase_repo.py --probe-readme
bash exp25_vor_or_preference_data/scripts/launch_effecterase_transfer_hal.sh
bash exp25_vor_or_preference_data/scripts/status_effecterase_transfer.sh
```
# Selective OR Data Tooling

Lightweight archive continuity and byte-count audit:

```bash
python exp25_vor_or_preference_data/scripts/inspect_vor_archives.py \
  --archive-dir /mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/downloads/fa09dc61128ca0418a4a13364d97a08018ea9cc7
```

Probe split gzip/tar readability without scanning the whole archive:

```bash
python exp25_vor_or_preference_data/scripts/inspect_vor_archives.py \
  --archive-dir /mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/downloads/fa09dc61128ca0418a4a13364d97a08018ea9cc7 \
  --stream-probe --probe-members 20
```

Index VOR-Eval members:

```bash
python exp25_vor_or_preference_data/scripts/index_vor_archive_members.py \
  --archive-dir /mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/downloads/fa09dc61128ca0418a4a13364d97a08018ea9cc7 \
  --groups VOR-Eval
```

Selective extraction example:

```bash
python exp25_vor_or_preference_data/scripts/safe_extract_vor_subset.py \
  --archive-dir /mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/downloads/fa09dc61128ca0418a4a13364d97a08018ea9cc7 \
  --groups VOR-Train VOR-Train-MASK \
  --sample-ids exp25_vor_or_preference_data/manifests/sample_ids_gate128.txt \
  --output-root /mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/extracted/exp25_gate128
```

## 2026-06-25 Post-Permission Asset Smoke

Executed on PAI as `hj` from runtime clone:

```bash
CUDA_VISIBLE_DEVICES=1 python /home/hj/runtime_code/H20_Video_inpainting_DPO_exp25_vor_run/exp25_vor_or_preference_data/scripts/run_vor_or_model_smoke.py \
  --model diffueraser \
  --manifest /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_vor_or_preference_data/root_cause_matrix_20260625/root_cause_sample12_manifest.jsonl \
  --project-root /home/hj/runtime_code/H20_Video_inpainting_DPO_exp25_vor_run \
  --output-root /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_vor_or_preference_data/postpermission_asset_smoke_20260625_130502 \
  --limit 1 \
  --num-frames 24 \
  --width 512 \
  --height 288 \
  --diffueraser-path /mnt/nas/hj/weights/diffuEraser/converted_weights_step48000 \
  --pcm-mode none \
  --prior-mode propainter \
  --no-pcm-steps 6 \
  --no-pcm-guidance 0.0 \
  --mask-dilation-iter 0 \
  --seed 20260625
```
