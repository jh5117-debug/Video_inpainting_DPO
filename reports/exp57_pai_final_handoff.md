# Exp57 PAI Adaptive Transition Final Handoff

Status: `EXP57_PAI_ONESTEP_NEGATIVE`

Branch: `research/exp57-void-adaptive-transition-pai-20260701`

Core base: `83e1bc7a5e5da231251d9c74f33a2ec49c8319f4`

## Scope

PAI ran one-step only on Q2/T500/S0 using the shared Exp57 adaptive transition-safe implementation. No 10-step, long training, VOR-Eval, hard comp, shared trainer edit, VOID official source edit, or `inference/metrics.py` edit was performed.

Because PAI could not create the requested NAS experiment output directory under `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo`, the lane used writable local output root:

`/home/hj/exp57_void_adaptive_transition_pai_outputs`

Weights and source data were still read from the official PAI/NAS asset roots. The missing Exp52 Q2 cache and Exp51 Q2 quadmask ablation files were relayed from H20 into PAI-local cache only.

## Cells

| Cell | Status | Full PSNR delta | Object PSNR delta | Overlap PSNR delta | Affected PSNR delta | Boundary PSNR delta | Outside PSNR delta | Visual | Selected scale | Lambda loser |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| ATS_SDPO_Q2_T500_S0 | NEGATIVE | 0.039160 | -0.337918 | -0.255698 | 0.108109 | -0.049336 | 0.075966 | 0 better / 0 tie / 4 worse | 1.0000 | 0.046926 |
| ATS_LINEAR_Q2_T500_S0 | NEGATIVE | 0.035889 | -0.003580 | -0.395469 | 0.053643 | -0.085521 | 0 better / 0 tie / 4 worse | 1.0000 | 0.018770 |

Best PAI diagnostic cell: `ATS_SDPO_Q2_T500_S0`.

## Required Questions

1. Did SDPO-style adaptive lambda help?

It produced full/outside/affected metric gains, but object and transition regions still failed the gate and visual review was `0/0/4`. It is not a PASS.

2. Did LinearDPO + adaptive transition safety help?

It nearly preserved object PSNR, but overlap and boundary regressed and visual review was `0/0/4`. It is not a PASS.

3. Did any cell preserve overlap / affected / boundary?

No. Affected improved on average, but overlap and boundary still regressed, and the visual evidence stayed worse on all heldout rows.

4. Which PAI cell is best?

`ATS_SDPO_Q2_T500_S0` by aggregate ranking, but only as a negative diagnostic.

5. Is any PAI cell one-step PASS?

No.

6. Should aggregator consider PAI candidate?

Yes, for failure-pattern comparison only. It should not unlock 10-step.

## Decision

`EXP57_PAI_ONESTEP_NEGATIVE`

VOID remains a VOR-OR inference baseline, same-model loser-generator candidate, and adapter-engineering candidate. This PAI lane does not provide third-backbone adapter evidence.
