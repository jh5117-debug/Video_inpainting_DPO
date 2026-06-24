# Exp27 Nontrivial Parity and LocalDPO Smoke - 2026-06-24

Status: `NONTRIVIAL_PARITY_AND_LOCALDPO_SMOKE_PASSED`

Controller run:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260624_three_lane_retry2`

SDPO:

- Real-batch SDPO parity: passed.
- Nontrivial gradient conflict case: passed.
- `lambda_safe < 1`: true.
- observed lambda: `0.314453125`.
- expected lambda: approximately `0.315`.
- conflict objective: `-0.0860637724`.
- conflict grad norm: `0.0348310508`.

Linear-DPO Frozen / EMA:

- Single-step parity: passed.
- Single-step loss: `-0.0407861434`.
- Single-step grad norm: `0.5717411041`.
- Frozen/EMA update max abs diff: `0.0`.
- Multi-step parity: passed for `5` steps.
- Multi-step losses: `[0.5547903776, -0.0017418617, 0.1638947427, -0.0091753993, -0.0009858343]`.
- Multi-step EMA max abs diff: `0.0`.

LocalDPO:

- Six-video corruption pair smoke: passed.
- Pair shape: `[6, 13, 4, 48, 80]`.
- Original region-aware loss smoke: one-step and ten-step both finite with non-zero gradients.
- One-step loss: `0.5581414104`.
- Ten-step first loss: `0.5038509369`.
- Ten-step last loss: `0.4880897403`.
- RC-FPO was not started.

Known limitation:

The official LocalDPO mask-generation file was not available at
`/home/hj/video_dpo_paper_code_cache/repos/Local-DPO/innerT2V/utils/random_mask_gen.py`,
so the official mask digest is marked `blocked_official_code_missing`. The
current LocalDPO result is a plumbing/original-loss smoke, not an official
LocalDPO reproduction.

Outputs:

- SDPO parity: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260624_three_lane_retry2/exp27_sdpo_nontrivial_real_batch_parity/real_batch_sdpo_parity.json`
- Linear parity: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260624_three_lane_retry2/exp27_linear_multistep_real_batch_parity/real_batch_linear_parity.json`
- LocalDPO smoke: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260624_three_lane_retry2/exp27_localdpo_six_video_smoke/localdpo_six_video_smoke.json`

Decision:

Exp27 completed the requested SDPO conflict parity, Linear Frozen/EMA multi-step parity, and LocalDPO 6-video smoke. No RC-FPO, long study, or training was started.
