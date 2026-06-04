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
