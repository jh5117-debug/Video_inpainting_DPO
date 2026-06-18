# Exp19b Exploratory 2000 Status

- status: completed negative / no-op ablation
- start checkpoint: Exp19b Stage2-500 flow adapter
- continuation: 1500 additional steps, total adapter steps = 2000
- loss: Exp11 DPO loss only, lambda_warp = 0
- eval target: DAVIS50
- training result: completed on PAI, checkpoints at 500 / 1000 / 1500 continuation steps and `last_weights`
- eval result: DAVIS50 completed
- decision: do not continue this setup to longer training; current best remains Exp11 outer b0.75 S2

The DAVIS50 evaluator label still prints `Exp19b_stage2_500`, but the eval
script points `EXP19_ADAPTER` to the exploratory 2000 adapter:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/exp19b_boundary_flow_adapter_s2_2000_exploratory_from500_limit100/last_weights/flow_adapter.pt
```
