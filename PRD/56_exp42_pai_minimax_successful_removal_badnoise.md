# PRD 56: Exp42 PAI MiniMax Successful-Removal + Bad-Noise Data Breakthrough

## Scope

Exp42 is an isolated PAI-side MiniMax data-signal experiment. It does not
repeat Exp38/Exp40 recipes and does not start blind long training. The goal is
to mine MiniMax-native successful-removal and failure cases under the audited
official protocol, then build success-vs-failure bad-noise states and
Stage2-style distillation/preference data before any short SFT/DPO gate.

## Branch and Roots

- Branch: `research/exp42-pai-minimax-successful-removal-badnoise-20260629`
- Base: `origin/research/exp40-minimax-psnr-safe-rescue-20260628`
- HAL worktree: `/home/hj/H20_Video_inpainting_DPO_exp42_pai_minimax_data`
- PAI code root: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp42_pai_minimax_data`
- Output root: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp42_pai_minimax_successful_removal_badnoise`
- Log root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp42_pai_minimax_successful_removal_badnoise`
- Runtime root: `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp42_pai_minimax_successful_removal_badnoise`

## Hard Boundaries

- Do not touch H20 worktrees, tasks, GPUs, outputs, or Exp41 artifacts.
- Do not modify `inference/metrics.py`, shared trainers, MiniMax official repo
  source, or prior Exp30/35/36/37/38/40/41 outputs.
- Do not use VOR-Eval for training, selection, or tuning.
- Do not use hard comp to fabricate quality gains; raw output remains primary.
- Do not run 500/1000/2000-step training from this prompt.
- Do not write `UNIVERSAL_ADAPTER`, `FINAL_SOTA`, or
  `TOP_CONFERENCE_NOVELTY_CONFIRMED`.

## Milestone Gates

1. Readback must be committed and pushed before GPU work.
2. Official MiniMax mining must use VOR-Train-derived sources only and the
   audited executable protocol: `UniPCMultistepScheduler`, fp16,
   `num_inference_steps=12`, `iterations=6`, no CFG, raw output primary.
3. Successful-removal pool gate requires at least 24 successful candidates and
   at least 24 medium-hard failures, technical-valid rate >=95%, and low
   outside damage.
4. Bad-noise v3 gate requires at least 24 usable success/failure pairs,
   bounded outside/winner risk, and hard-state local/random gradient ratio
   >=1.5.
5. Stage2 data gate requires train >=32, search >=16, shadow >=16, scene-group
   disjoint splits, decode pass, non-empty masks, and no VOR-Eval leakage.
6. Short SFT/DPO gates remain locked until the corresponding data gates pass.

## 2026-06-29 Readback

Status: `EXP42_PAI_MINIMAX_DATA_READBACK_COMPLETED`.

Start HEAD: `7dd81ef8baf1377009a4e74b022b9904e2a84957`.

Readback confirms:

- MiniMax is plumbing-positive and protocol-audited, but not quality-positive.
- Exp36 ruled out ignored-weight and checkpoint-load failure: identity replay
  MAE `0.0`; 1.01x perturbation full/mask MAE `0.088218` / `0.156302`.
- Exp36 winner-SFT lowered train loss and moved outputs, but heldout visual
  better rows were `0/24`.
- Exp37 LocalDPO-badnoise R1/R2/R3 reached only `1/16` better rows per recipe.
- Exp38 R1 had weak raw metric movement but boundary/outside and visual gates
  failed: full/mask/boundary/outside `+0.102167` / `+0.117230` /
  `-0.141510` / `-0.037262`.
- Exp40 PSNR-safe SFT grid was negative; best aggregate recipe still had
  full/mask/boundary/outside `-1.816781` / `-1.634597` / `-1.899575` /
  `-2.624405`.

PAI GPU0/GPU1 readback at hostname `dsw-753014-85f54df947-bkp7h` found both
cards at `0 MiB`, `0%`, with no compute PID. No PID was killed in readback.

Available sources:

- Exp30 Gate64 V3 train32/heldout16.
- Exp37 LocalDPO-style train32/heldout16.
- Exp38 LocalDPO v2 train30/heldout13 filtered and bad-noise v2 states.
- Exp40 LocalDPO v3 minimum pool train64/search24/shadow24.
- PAI official MiniMax repo and weights are present:
  `/mnt/nas/hj/official_repos/MiniMax-Remover_28e12b4` and
  `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/minimax_remover/current`.

Exp41 H20 PRD/report files are not present on this Exp40-based branch and were
not read from the protected H20 worktree. This is recorded instead of being
fabricated.

Reports:

- `reports/exp42_pai_minimax_data_readback.md`

Next status target: `MINIMAX_SUCCESSFUL_REMOVAL_POOL_READY`.
