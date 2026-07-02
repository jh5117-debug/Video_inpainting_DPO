# Exp60B VPData Availability Audit

Status: `EXP60B_VPDATA_AVAILABLE`

## Sources Checked

- Official repo docs: `/home/hj/dpo-2-1-exp/third_party_baselines/VideoPainter/README.md`
- Official download script:
  `/home/hj/dpo-2-1-exp/third_party_baselines/VideoPainter/data_utils/VPData_download.py`
- Hugging Face dataset: `https://huggingface.co/datasets/TencentARC/VPData`
- Hugging Face API tree:
  `https://huggingface.co/api/datasets/TencentARC/VPData/tree/main?recursive=1&expand=1`

## Findings

VPData is available as a public Hugging Face dataset. The public dataset page
reports:

- 392,077 rows.
- Total file size about 1.87 TB.
- 390K+ mask sequences and video captions.
- Files include `pexels.csv`, `videovo.csv`, Pexels mask zip groups, VideoVo
  mask zip groups, and VideoVo raw-video zip groups.

The official README says Pexels raw videos are not all bundled directly and are
downloaded by `data_utils/VPData_download.py`. The local script loops over all
rows in `pexels.csv`, so it must not be run unmodified for Exp60B.

## Selective Download Feasibility

Selective download is feasible in principle but needs a custom script:

1. Download only metadata CSV files first.
2. Build deterministic train1000/test100 row lists using seed `20260702`.
3. For VideoVo rows, download only required raw-video/mask zip shards.
4. For Pexels rows, download only selected URLs from `pexels.csv`.
5. Verify no overlap by video id/source id/scene id before transfer.

## Blockers

The current HAL session cannot reach H20 via a configured alias or recovered
host. H20 storage and download cannot be performed yet.

Status for download: `EXP60B_H20_DOWNLOAD_BLOCKED_CONNECTIVITY`.

