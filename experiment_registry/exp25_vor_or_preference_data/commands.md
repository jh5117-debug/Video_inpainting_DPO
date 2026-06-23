# Exp25 Commands

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
