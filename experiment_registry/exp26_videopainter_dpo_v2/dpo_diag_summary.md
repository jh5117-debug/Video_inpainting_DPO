# Exp26 DPO Diagnostic Summary

## Primary-32 10-Step Gate

- status: `VIDEOPAINTER_10STEP_GATE_PASSED`
- rows: `10`
- final loss: `2.091540575027466`
- final DPO loss: `1.7013121843338013`
- final implicit accuracy: `0.0`
- final winner improvement mean: `0.0`
- final loser degradation mean: `0.3309490978717804`
- final loser-dominant ratio: `0.0`
- max grad norm: `170.04361478093455`
- NaN/Inf count: `0`
- reference gradients: frozen in L0/L1 and strict preflight checks
- strict reload: checkpoint-1 and checkpoint-10 passed

## Primary-32 50-Step Gate

- status: `VIDEOPAINTER_ADAPTER_POSITIVE`
- rows: `50`
- final total loss: `0.9509385228157043`
- final DPO loss: `0.8550137281417847`
- final implicit accuracy: `0.0`
- final grad norm: `30.821309345579444`
- max grad norm: `471.68358081969296`
- p95 grad norm: `136.25990432375542`
- final loser-dominant ratio: `0.0`
- NaN/Inf count: `0`
- strict reload/preflight: checkpoint-10/20/30/40/50 passed
