# Exp50 VOID Adapter 10-Step V2

Status: `VOID_ADAPTER_10STEP_NEGATIVE`

## Gate Decision

10 optimizer steps completed on H20 with finite loss, finite heldout outputs, strict step10 reload, no VOR-Eval, and no hard comp. The gate is negative, not blocked: the run finished technically, but it does not meet the 10-step promising/positive criteria.

Reasons:

- Mean full PSNR delta: -0.000965
- Mean outside PSNR delta: 0.043422
- Mean mask PSNR delta: -0.229878
- Mean affected PSNR delta: 0.019341
- Mean boundary PSNR delta: -0.063034
- Local/effect mean improvements: 1 / required >= 2
- Metric-worse samples: 3 / 4
- Codex visual better/tie/worse: 0/3/1

## Per-Sample

- BLENDER_CON001_00742: TEN_STEP_METRIC_WORSE; visual=tie; d_full=-0.057777; d_out=-0.051460; d_mask=-0.324869; d_aff=0.214595; d_boundary=0.058123
- BLENDER_CON001_00744: TEN_STEP_METRIC_WORSE; visual=tie; d_full=-0.095502; d_out=-0.088410; d_mask=-0.146602; d_aff=0.055233; d_boundary=0.030273
- REAL_ENV102_00001_002_02: TEN_STEP_LOCAL_OR_FULL_IMPROVE; visual=tie; d_full=0.245445; d_out=0.413160; d_mask=-0.240319; d_aff=-0.109277; d_boundary=-0.244061
- REAL_ENV200_00001_006_02: TEN_STEP_METRIC_WORSE; visual=worse; d_full=-0.096026; d_out=-0.099603; d_mask=-0.207721; d_aff=-0.083188; d_boundary=-0.096472

## Visual Review

Codex opened the temporal strip, object crop, outside crop, and temporal-diff heatmap evidence for all four heldout samples. The outputs are finite and visually close to Step0, with no collapse, no systematic outside damage, and no systematic tone drift. However, the visual review does not show a clear heldout improvement, and one sample shows mild local/texture worsening.

## Safety

10 optimizer steps only. No VOR-Eval, no hard comp, no 30/50/100/300/500-step run. LPIPS/Ewarp/TC were unavailable on the H20 relay path and are not used for a positive claim.

## Scientific Conclusion

VOID remains usable as a VOR-OR inference baseline and same-model loser generator candidate. The 10-step adapter micro gate is negative, so VOID is not third-backbone adapter evidence.
