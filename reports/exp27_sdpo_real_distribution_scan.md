# Exp27 SDPO Real Distribution Scan

Status: `SDPO_REAL_RESIDUAL_PROXY_SCAN_COMPLETE`

Run root:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_26_27_followup_20260624/exp27_sdpo_real_distribution_scan`

Important caveat:

This is a real-video residual proxy scan, not a full DiffuEraser policy-forward
gradient scan. It uses real preference rows and real condition/candidate/winner
frames and does not manually construct conflict gradients, but it does not load
the policy/reference models. It cannot promote RC-FPO by itself.

Inputs:

- Rows: first 32 real Gate32 VOR preference rows.
- Timesteps/frames per row: 4.
- Records: 128.
- Winner: `V_bg`.
- Condition: `V_obj`.
- Loser proxy: existing DiffuEraser Gate32 raw OR candidate.

Results:

| metric | value |
| --- | ---: |
| records | 128 |
| lambda < 1 ratio | 0.4453125 |
| lambda min | 0.2246925 |
| lambda mean | 0.8942396 |
| lambda max | 1.0 |
| unsafe tiny-step winner-change rate | 0.0 |

Interpretation:

The residual-proxy distribution contains many nontrivial SDPO safe-lambda cases:
`44.5%` of records have `lambda_safe < 1`. This supports continuing a true
policy-forward distribution scan, but it is not enough to start RC-FPO or any
objective study.

Outputs:

- CSV: `reports/exp27_sdpo_real_distribution_scan.csv`
- Summary JSON: `reports/exp27_sdpo_real_distribution_scan_summary.json`
- PAI output: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_26_27_followup_20260624/exp27_sdpo_real_distribution_scan`
