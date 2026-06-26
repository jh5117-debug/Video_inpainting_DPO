# Exp29 OR Adapter Readback

Date: 2026-06-26

## Git

- Branch: `research/exp29-minimax-effecterase-adapter-feasibility-20260626`
- Base branch: `origin/research/exp26-videopainter-dpo-v2`
- Initial HEAD: `568a7dfb48bcdfce893176a1dd48c653414a13a8`
- Worktree: `/home/hj/H20_Video_inpainting_DPO_exp29_or_adapters`

## Files Read

- `PRD/00_current_status.md`
- `PRD/01_experiment_matrix.md`
- `PRD/48_exp26_videopainter_dpo_v2.md`
- `reports/exp26_vp_shadowdev_final_decision.md`
- `reports/exp26_external_validation_visual_review.md`
- `reports/exp26_videopainter_result_pack.md`
- `reports/exp26_third_model_compatibility_audit.md`
- `reports/exp26_third_model_next_adapter_recommendation.md`
- Exp26 VideoPainter code and scripts inventory

## Exp26 Source State

VideoPainter is confirmed on locked VOR-BG search-dev and independent
shadow-dev. The accepted statement is
`CROSS_BACKBONE_ADAPTER_EVIDENCE_CONFIRMED` because DiffuEraser and
VideoPainter both have successful adapter evidence.

The external DAVIS-derived 49F validation did not confirm Step50. The visual
review reported Step50 slightly better in 3 rows, tied in 5 rows, and worse in
24 rows, with 29 rows showing Step50-specific local artifacts. This blocks
universal-adapter, final-SOTA, and top-conference novelty claims.

## Why MiniMax

MiniMax-Remover is OR-native and has a local repo candidate with pipeline and
transformer files. It is the strongest current candidate for an OR baseline and
could become a third true adapter only if verified weights, flow-matching
target, trainable forward, policy/reference zero-gap, strict reload, and
micro-gate evidence pass.

## Why EffectErase

EffectErase is a strong OR diagnostic and baseline candidate, but it is
VOR-related/VOR-trained according to the current Exp26 compatibility audit. It
must not be treated as primary on-policy VOR loser evidence. True adapter claims
require non-confounded data and trainable-forward evidence.

## Local Asset Hints

- MiniMax repo hint:
  `/home/hj/dpo-2-1-exp/third_party_baselines/MiniMax-Remover/repo`
- MiniMax HF cache hint:
  `/home/hj/.cache/huggingface/hub/models--zibojia--minimax-remover`
- EffectErase repo hint:
  `/home/hj/video_inpainting_third_party/EffectErase`

These hints require full audit before any smoke.

## Left CLI Protection Readback

PAI read-only audit saw left CLI runtime files under
`/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/cli4`. Active left-side Exp28
processes were using GPU3 and GPU4 with process name `Exp28CLI4`. Runtime locks
also reserved GPU1/GPU2 lanes. No signal was sent and no left-side file was
modified.

## This Round Restrictions

- No long training.
- No VideoPainter 100-step continuation.
- No RC-FPO.
- No third-backbone DPO training until feasibility gates pass.
- No VOR-Eval training, threshold, or checkpoint selection.
- No universal-adapter claim.

## Planned Outputs

- `reports/exp29_minimax_repo_weight_audit.md`
- `reports/exp29_minimax_repo_weight_audit.csv`
- `reports/exp29_minimax_asset_matrix.json`
- `reports/exp29_effecterase_repo_weight_audit.md`
- `reports/exp29_effecterase_repo_weight_audit.csv`
- `reports/exp29_effecterase_asset_matrix.json`
- `reports/exp29_minimax_effecterase_adapter_summary.md`
- `reports/exp29_minimax_effecterase_adapter_summary.csv`

