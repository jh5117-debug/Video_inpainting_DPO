# PRD 48: Exp32 DiffuEraser VOR-OR 2000-Step

Date: 2026-06-27

## Objective

Create an isolated DiffuEraser VOR-OR lane that can run Stage1/Stage2
2000-step training only after a strict data gate passes.

## Isolation

- Branch: `research/exp32-diffueraser-vor-or-2000step-20260627`
- Base: `origin/research/exp25-vor-or-preference-data`
- Base HEAD: `bc6dc80206f5e397bda577ba62f9371813e5a657`
- HAL worktree: `/home/hj/H20_Video_inpainting_DPO_exp32_diffueraser_vor_or`
- PAI runtime root:
  `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp32_diffueraser_vor_or`
- Experiment output root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp32_diffueraser_vor_or_2000step`
- Log root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp32_diffueraser_vor_or_2000step`

Exp32 must not write to Exp25 CLI4 outputs, Exp30/MiniMax outputs,
`inference/metrics.py`, or shared trainer code.

## Prior Readback

Exp25 base status:

- VOR core download complete and verified.
- VOR-Eval extracted as held-out only, 43 aligned triplets.
- Gate32 canonical DiffuEraser raw generation completed but had poor quality
  yield: `medium-hard=11`, `trivial-bad=21`, `technical-invalid=0`.
- Root-cause matrix selected the useful DE-B stack after the base branch HEAD.

CLI4 read-only readback from the isolated Exp25 continuation:

- report: `/home/hj/H20_Video_inpainting_DPO_exp25_cli4/reports/exp25_gate16_deb_cli4_result.md`
- status: `EXP25_DIFFUSERASER_GATE16_PASSED`
- fixed stack: `DE-B_sft_raw6_d8_propainter`
- `pcm_mode=none`
- `prior_mode=propainter`
- `no_pcm_steps=6`
- `guidance=0.0`
- `mask_dilation_iter=8`
- `hard_comp=false`
- condition: `V_obj`
- winner: `V_bg`
- loser: DiffuEraser raw output
- technical valid: `16/16`
- medium-hard eligible: `7/16`
- hard but plausible: `7/16`
- trivial bad: `2/16`
- technical invalid: `0/16`

This is a useful Gate16, but it is not a train32 + heldout16 scene-disjoint
data gate by itself.

## Training Gate

Stage1 2000-step training is blocked until one of these is true:

1. VOR-OR multi-model pool exists with at least train32 + heldout16
   scene-disjoint pairs.
2. DiffuEraser-only DE-B Gate64 exists with at least train32 + heldout16
   scene-disjoint pairs, technical valid >= 48, medium-hard/hard-plausible
   >= 40, trivial-bad <= 8.
3. Controlled LocalDPO-style OR corruption Gate64 exists with train32 +
   heldout16 scene-disjoint pairs and outside preservation passed.

Current readback status: none of these gates has been verified in this branch.

## Right-Side Protection

Read-only PAI checks found no active compute process, but Exp30/MiniMax outputs
exist and stale MiniMax candidate locks reserve GPU0 and GPU5. Exp32 may use
GPU2 for source-pool or candidate-generation smoke only after its own lock and
preflight pass. It must not start MiniMax training or write Exp30 outputs.

## Status

Current status: `EXP32_READBACK_COMPLETED_DATA_GATE_PENDING`

Training status: `DIFFUSERASER_VOR_OR_BLOCKED`

