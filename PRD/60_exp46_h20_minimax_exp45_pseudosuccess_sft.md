# Exp46 H20 MiniMax Exp45 Pseudo-Success SFT Validation

Status: EXP45_H20_MANIFESTS_READY

Branch: `research/exp46-h20-minimax-exp45-pseudosuccess-sft-20260629`
Start HEAD: `feef1b73317bea903e0e247d077d84c740665fa4`
Exp45 source branch: `origin/research/exp45-pai-minimax-pair-scaleup-20260629`
Exp45 source HEAD: `d0c8430a5ba35f37415ed52d53040829ef1123d6`

## Scope

Exp46 is an H20-only validation of the PAI Exp45 formal Stage2 handoff package. The first training path is pseudo-success SFT only. GT-only SFT and DPO are explicitly out of scope for the first gate.

## Milestone Status

- A readback: complete
- B mirror Exp45 required files: complete (`EXP45_H20_MIRROR_READY`)
- C rewrite/validate manifests: complete (`EXP45_H20_MANIFESTS_READY`)
- D BF16/environment preflight: pending
- E Step0 baseline: pending
- F pseudo-success SFT 30-step: pending
- G pseudo-success SFT 100-step: conditional only if 30-step promising
- H decision/paper positioning: pending

## Guardrails

No PAI writes, no PAI GPU, no GT-only SFT, no DPO, no long training, no VOR-Eval, no hard comp, no shared trainer or metrics edits, no MiniMax positive claim before real shadow pass.


## Milestone B Mirror Validation

Status: `EXP45_H20_MIRROR_READY`

- H20 mirror root: `/home/nvme01/H20_Video_inpainting_DPO/data/h20_mirror/minimax/pai_abs`
- Exp45 source HEAD: `d0c8430a5ba35f37415ed52d53040829ef1123d6`
- Required paths: `326`
- Required files: `232`
- Required directories: `94`
- Missing paths: `0`
- SHA mismatches: `0`
- Required-path mirrored bytes: `359462035`

The mirror was completed by H20 reading PAI/NAS through SSH agent forwarding. PAI was read-only, PAI GPUs were not used, and no training or optimizer step occurred. Manifest rewrite/validation remains the next gate.


## Milestone C Manifest Rewrite Validation

Status: `EXP45_H20_MANIFESTS_READY`

- H20 mirror root: `/home/nvme01/H20_Video_inpainting_DPO/data/h20_mirror/minimax/pai_abs`
- Pseudo-success split: `64/24/24`
- GT distillation split: `64/24/24`
- Preference split: `64/24/24`
- Failed validation rows: `0`
- Scene overlap OK: `True`
- VOR-Eval rows: `0`
- Hard-comp rows: `0`
- Bad-noise required/unmatched: `0/0`
- MP4 fallback decode rows: `128`

Manifests are H20-local and preserve original PAI paths only in `pai_*` fields. This milestone did not run training or optimizer steps.
