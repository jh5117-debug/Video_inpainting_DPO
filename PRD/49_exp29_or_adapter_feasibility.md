# PRD 49: Exp29 OR Adapter Feasibility

## Objective

Exp29 audits MiniMax-Remover and EffectErase as:

1. OR baselines;
2. loser generators;
3. possible future true DPO adapter backbones.

This is a feasibility and evidence-quality track. It does not start long
training, RC-FPO, VideoPainter 100-step continuation, or third-backbone DPO
training.

## Source State

Exp26 VideoPainter LoVI-DPO completed search-dev, independent shadow-dev, and
external DAVIS-derived validation. The accepted statement is:

`CROSS_BACKBONE_ADAPTER_EVIDENCE_CONFIRMED`

The forbidden statement remains:

`UNIVERSAL_ADAPTER`

External DAVIS-derived VideoPainter validation was not confirmed, so cross-data
robustness remains open.

## Left CLI Protection

The left CLI controller and its worktrees are treated as read-only external
state. Exp29 may inspect GPU/process status but must not modify files, locks,
heartbeats, branches, worktrees, or processes associated with:

- `/home/hj/cli4_controller`
- `/home/hj/H20_Video_inpainting_DPO_exp25_cli4`
- `/home/hj/H20_Video_inpainting_DPO_exp27_cli4`
- `/home/hj/H20_Video_inpainting_DPO_exp28_inner_boundary`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/cli4`

## Initial Readback

- Branch: `research/exp29-minimax-effecterase-adapter-feasibility-20260626`
- Base: `origin/research/exp26-videopainter-dpo-v2`
- Initial base HEAD: `568a7dfb48bcdfce893176a1dd48c653414a13a8`
- PRD selected: `PRD/49_exp29_or_adapter_feasibility.md`
- New isolated code root: `exp29_or_adapter_feasibility/`
- New registry: `experiment_registry/exp29_or_adapter_feasibility/`

## Initial Local Asset Hints

- MiniMax local candidate repo:
  `/home/hj/dpo-2-1-exp/third_party_baselines/MiniMax-Remover/repo`
- EffectErase local candidate repo:
  `/home/hj/video_inpainting_third_party/EffectErase`

These are hints only. Exp29 must still perform repo, license, weight, code, and
trainable-forward audits before running any smoke.

## Promotion Gates

MiniMax can only become `MINIMAX_TRUE_ADAPTER_FEASIBILITY_CONFIRMED` after:

- verified repo and weights;
- official inference smoke with video review;
- native trainable forward;
- policy/reference zero-gap;
- one-step strict reload;
- 10-step micro gate with held-out review.

EffectErase can only become `EFFECTERASE_TRUE_ADAPTER_FEASIBILITY_CONFIRMED`
after the same technical evidence plus a non-confounded data design. If its
available evidence is VOR-trained only, it must be labeled technical or
baseline/diagnostic rather than scientific positive.

