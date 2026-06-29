# Exp46 H20 MiniMax Exp45 Pseudo-Success SFT Validation

Status: EXP46_READBACK_READY

Branch: `research/exp46-h20-minimax-exp45-pseudosuccess-sft-20260629`
Start HEAD: `feef1b73317bea903e0e247d077d84c740665fa4`
Exp45 source branch: `origin/research/exp45-pai-minimax-pair-scaleup-20260629`
Exp45 source HEAD: `d0c8430a5ba35f37415ed52d53040829ef1123d6`

## Scope

Exp46 is an H20-only validation of the PAI Exp45 formal Stage2 handoff package. The first training path is pseudo-success SFT only. GT-only SFT and DPO are explicitly out of scope for the first gate.

## Milestone Status

- A readback: complete
- B mirror Exp45 required files: pending
- C rewrite/validate manifests: pending
- D BF16/environment preflight: pending
- E Step0 baseline: pending
- F pseudo-success SFT 30-step: pending
- G pseudo-success SFT 100-step: conditional only if 30-step promising
- H decision/paper positioning: pending

## Guardrails

No PAI writes, no PAI GPU, no GT-only SFT, no DPO, no long training, no VOR-Eval, no hard comp, no shared trainer or metrics edits, no MiniMax positive claim before real shadow pass.
