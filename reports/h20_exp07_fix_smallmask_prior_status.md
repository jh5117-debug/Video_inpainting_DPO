# H20 Exp07 Fix Smallmask Prior Status

Updated: 2026-06-05 CST

## What succeeded

- H20 SSH succeeded intermittently.
- Registry was created before any launch:
  `/home/nvme01/H20_Video_inpainting_DPO/experiment_registry/exp07_fix_smallmask_prior/config.yaml`
- H20 SFT-48000 DiffuEraser weights exist:
  `/home/nvme01/H20_Video_inpainting_DPO/weights/diffuEraser/converted_weights_step48000`
- GPU snapshot during registry/precheck:

```text
0, 65, 97871, 27
1, 1, 97871, 0
2, 1, 97871, 0
3, 1, 97871, 0
4, 1, 97871, 0
5, 1, 97871, 0
6, 1, 97871, 0
7, 1, 97871, 0
```

## What is blocked

- Data root was not present in the registry precheck output:
  `data/generated_losers/exp07_fix_videodpo_smallmask15_20_prior_k4`
- A dry-run preflight for the VideoDPO smallmask generation command could not be completed because H20 SSH was flapping / resetting connections.
- No data generation was started.
- No training was started.

## Next safe step

Run only the preflight first. If it confirms the VideoDPO train yaml, ProPainter weights, smallmask config, and dry-run are valid, then start smallmask data generation. Only after `selected_primary_comp.repaired.jsonl` exists should the Stage1 gate launcher run.
## 2026-06-05 CST Update: data generation launched

- Data generation was launched on H20 after dry-run passed.
- PID: `2590851`
- Log: `/home/nvme01/H20_Video_inpainting_DPO/logs/pipelines/exp07_fix_smallmask_prior_data_generation_h20_20260605_050336.log`
- PID file: `/home/nvme01/H20_Video_inpainting_DPO/logs/pipelines/exp07_fix_smallmask_prior_data_generation_h20.pid`
- Output root: `/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/exp07_fix_videodpo_smallmask15_20_prior_k4`
- GPU: `CUDA_VISIBLE_DEVICES=1`, `DIFFUERASER_GPU=1`
- Scope: `limit=1000`, `models=diffueraser`, `K=4`, `mask_area=0.15-0.20`, `prior=propainter` via DiffuEraser wrapper.
- Stage1 training was not launched. It must wait for `manifests/selected_primary_comp.jsonl` or repaired equivalent.

Monitor snapshot at 05:04:46 CST:

```text
process = running
GPU1 memory = 4 MiB, util = 0 at snapshot (likely startup / CPU decode stage)
output candidate dirs count = 2
fatal errors = none observed yet; log had not flushed key lines
```

