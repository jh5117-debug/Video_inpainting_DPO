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

## 2026-06-05 10:21 CST monitor

- SSH with the new key succeeded.
- Data generation process is still running:
  `PID=2590851`
- Current command:
  `tools/videodpo_generated_loser_calibration.py --output_root data/generated_losers/exp07_fix_videodpo_smallmask15_20_prior_k4 --models diffueraser --limit 1000 ... --mask_policy_config configs/generation/videodpo_partialmask_policy_v2_smallmask15_20_k4.yaml ... --skip_existing`
- GPU snapshot:

```text
0, 28 MiB, 97871 MiB, 0%
1, 6566 MiB, 97871 MiB, 0%
2, 1 MiB, 97871 MiB, 0%
3, 1 MiB, 97871 MiB, 0%
4, 1 MiB, 97871 MiB, 0%
5, 1 MiB, 97871 MiB, 0%
6, 1 MiB, 97871 MiB, 0%
7, 1 MiB, 97871 MiB, 0%
```

- Output progress:
  - `candidates_all.jsonl`: 77 rows
  - file counts: 10,148 png, 156 mp4, 78 log, 77 json, 1 jsonl
  - latest observed file: `work/videodpo_pair000019/mask_001/diffueraser/run_or/sample/propainter.mp4`
- Selected manifests are not ready yet:
  `selected_primary_comp.jsonl` / `.repaired.jsonl` not present.
- Stage1 training remains blocked until a selected primary comp manifest exists.

## 2026-06-05 11:01 CST update: switched data generation to GPUs 0-5

- User requested H20 generation to use GPUs 0-5 instead of one GPU.
- Added and deployed sharded launcher:
  `scripts/launch_exp07_fix_smallmask_prior_data_generation_h20_gpus0_5.sh`
- The old single-GPU process was stopped safely:
  `PID=2590851`
- Its partial manifest was preserved:
  `manifests/candidates_all.single_gpu_partial_<timestamp>.jsonl`
- First sharded attempt failed before generation because `VIDEO_DPO_TRAIN_DATA_YAML`
  still pointed to a PAI `/mnt/nas/...` path. The launcher was fixed to export the
  H20-local train yaml explicitly.
- Current 0-5 sharded run:
  - PID: `503376`
  - Log: `logs/pipelines/exp07_fix_smallmask_prior_data_generation_h20_gpus0_5_20260605_110124.log`
  - Shards root:
    `data/generated_losers/exp07_fix_videodpo_smallmask15_20_prior_k4/_shards_gpu0_5_20260605_110124`
  - Pair range: `[0, 1000)`
  - GPUs: `0,1,2,3,4,5`
  - workers per GPU: `1`
  - shard size: `5`
  - model: `diffueraser`
  - mask policy: `videodpo_partialmask_policy_v2_smallmask15_20_k4`
  - train yaml:
    `/home/nvme01/H20_Video_inpainting_DPO/data/VideoDPO/configs/vc2_dpo/vidpro/train_data.absolute.yaml`
- Launch log confirmed six initial shards:

```text
[launch] shard_000000_000005 gpu=0
[launch] shard_000005_000010 gpu=1
[launch] shard_000010_000015 gpu=2
[launch] shard_000015_000020 gpu=3
[launch] shard_000020_000025 gpu=4
[launch] shard_000025_000030 gpu=5
```

- A later snapshot showed fresh shard files being written at 11:04 CST.
- Stage1 training is still not launched. It remains blocked until the sharded
  generator finishes merge/selection and writes `selected_primary_comp.jsonl`.
