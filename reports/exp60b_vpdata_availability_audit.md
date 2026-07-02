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

Selective download is feasible for a Pexels-only first subset and needs a
custom script:

1. Download only metadata CSV files first.
2. Build deterministic train1000/test100 row lists using seed `20260702`.
3. For Pexels rows, download only selected URLs from `pexels.csv`.
4. Exclude VideoVo rows for this first pass because VideoVo raw videos are
   bundled as multi-GB zip shards.
5. Verify no overlap by video id/source id/scene id before transfer.

Generated plan:

- `manifests/exp60b_vpdata_train1000_sources_h20.jsonl`
- `manifests/exp60b_vpdata_test100_sources_h20.jsonl`
- `reports/exp60b_vpdata_subset_plan.csv`
- `reports/exp60b_vpdata_subset_plan_summary.json`

Plan stats:

- train CSV rows: 390,509.
- train unique Pexels-eligible rows: 124,426.
- train selected: 1,000.
- test CSV rows: 568.
- test unique Pexels-eligible rows after excluding train source overlap: 430.
- test selected: 100.
- train/test overlap: 0.
- video downloads performed during plan generation: 0.

## Blockers

HAL direct H20 SSH is intermittent, but PAI relay to H20 is available and H20
storage passes the Exp60B hard-stop threshold. Download should be launched on
H20 via PAI relay after this tooling commit is pushed.

Status for download: `EXP60B_H20_READY_VIA_PAI_RELAY`.
