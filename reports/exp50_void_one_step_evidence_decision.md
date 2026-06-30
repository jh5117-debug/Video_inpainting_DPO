# Exp50 VOID One-Step Evidence Decision

Time: 2026-07-01T01:07:56+08:00

Final status: `VOID_ONE_STEP_VIDEO_EVIDENCE_MIXED`

## Decision

H4b successfully generated Step0/Step1 heldout4 video evidence, computed metrics, and Codex opened all four temporal evidence sheets. The result is mixed, not a pass. Therefore H5 10-step remained locked and was not run.

This is not a catastrophic failure: no collapse was observed, and two samples were visual ties. However the one-step update did not meet the conservative PASS criteria because the mean full PSNR delta was below the safety threshold and two heldout samples were metric/visual worse.

## Required Answers

1. Is VOID usable for VOR-OR inference? Yes. F2 status `VOID_INFERENCE_SMOKE_PASS`; technical valid 8 / 8.
2. Is VOID a baseline / loser generator? Yes. Gate8 classification counts: {'MEDIUM_HARD_LOSER': 2, 'TOO_CLOSE': 2, 'VOID_OUTPUT_USABLE': 4}.
3. Is VOID a true adapter candidate? Engineering candidate only; not quality-positive adapter evidence.
4. Did SFT parity pass/explain? `VOID_SFT_FORWARD_PARITY_EXPLAINED`; target parameterization `v_prediction`.
5. Did preference forward pass? `VOID_PREFERENCE_FORWARD_PASS`.
6. Did zero-gap pass? `VOID_ZERO_GAP_PASS`.
7. Did one-step video evidence pass? No. H4b status `VOID_ONE_STEP_PARETO_MIXED`.
8. Did 10-step run? No.
9. Did 10-step pass if run? Not applicable; H5 was not run.
10. Did VOID become third adapter evidence? No.
11. If not, exact blocker: `VOID_ONE_STEP_HELDOUT_MIXED_METRICS_VISUAL_NOT_PASS`.
12. Should we continue VOID, resume ROSE, or stop third-model search? Continue VOID only if the next experiment changes the update target or trainable subset; do not run 10-step from this one-step state. ROSE can resume in parallel as a separate feasibility track.

## H4b Evidence

- Checkpoint: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp50_pai_void_adapter_feasibility/one_step_v2/adapter_proj_out_step1.pt`
- Adapter SHA256: `849326121699e51673990df7bef52c578245812ad934a88bbe6f2acf93b972d5`
- Step1 merged checkpoint: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp50_pai_void_adapter_feasibility/one_step_heldout_evidence_v2/checkpoints/void_pass1_step1_proj_out.safetensors`
- Step1 merged SHA256: `d57efd25280baae896b8e4d396df3233cf1ac6411cb9f0d7cccdea5fd4dc4515`
- Step0 / Step1 videos: 4 / 4
- Requested GPUs: [0, 1]; root processes killed: []
- Mean full PSNR delta: -0.025049
- Mean outside PSNR delta: 0.028255
- Mean mask PSNR delta: -0.513424
- Visual better/tie/worse: 0 / 2 / 2

## Safety

No VOR-Eval was used. No hard comp was used. H4b did not run optimizer steps. H5 10-step was not run. No 30/50/100/300/500-step run was launched. VOID official repo source, shared trainer, and `inference/metrics.py` were not modified.
