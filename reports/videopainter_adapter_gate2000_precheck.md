# VideoPainter Adapter Gate2000 Precheck

Date: 2026-06-15

## Status

Trainer implemented after the original block. The 2000-step gate is still
**not** launched until PAI trainer preflight passes.

## User Request

The user explicitly requested skipping VideoPainter smoke1/smoke20 and going
directly to the 2000-step gate, but still required the minimum hard precheck:

- VideoPainter repo / training entry exists.
- Data exists.
- Weights exist.
- Frozen reference model can be constructed.
- Output directory is writable.
- GPU is available.
- Loss can be connected to policy/reference winner/loser losses.

## HAL Precheck

Passed:

- Local VideoPainter repo exists:
  `/home/hj/dpo-2-1-exp/third_party_baselines/VideoPainter`
- Repo URL:
  `https://github.com/TencentARC/VideoPainter`
- Commit:
  `bbab6cd5cd5cb89f0e2444305c32fd74a010ae0a`
- Upstream training/inference/eval entries exist:
  - `train/VideoPainter.sh`
  - `train/train_cogvideox_inpainting_i2v_video.py`
  - `infer/inpaint.py`
  - `evaluate/eval_inpainting.py`

Now implemented:

- Isolated adapter trainer:
  `exp14_adapter_videopainter/code/train_videopainter_dpo_adapter.py`
- Local static checks passed:
  - `python -m py_compile`
  - `bash -n` for the gate launcher
- The trainer defines:
  - `m_w`
  - `m_l`
  - `m_w_ref`
  - `m_l_ref`
  - region-local normalized-gap DPO loss
  - VideoPainter adapter diagnostics

Still pending:

- PAI has not yet run the trainer preflight with real VideoPainter weights.
- Gate2000 remains blocked until that preflight passes.

## PAI Precheck

PAI connection succeeded:

```text
host: dsw-753014-dc85766cb-4v2jj
```

Checked PAI repo roots:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp09_10_11_pai_sync
```

Findings:

- Data exists:
  - `/mnt/workspace/hj/nas_hj/data/external/youtubevos_432_240`
  - `/mnt/workspace/hj/nas_hj/data/external/youtubevos_432_240_eval100`
  - `/mnt/workspace/hj/nas_hj/data/external/davis_432_240`
  - `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4/manifests/selected_primary_comp.repaired.pai_paths.jsonl`
- GPUs are idle enough for a run:

```text
0, 0, 143771, 0
1, 0, 143771, 0
2, 0, 143771, 0
3, 0, 143771, 0
4, 244, 143771, 0
5, 4, 143771, 0
6, 292, 143771, 0
7, 58071, 143771, 0
```

Original PAI failure:

- PAI checked repos did not contain `exp14_adapter_videopainter`.
- PAI checked repos did not contain the adapter trainer.
- The VideoPainter repo was not found in the checked project paths.

Required before the next PAI run:

- sync this Exp14 folder to PAI;
- place or verify VideoPainter repo / base model / branch checkpoint paths;
- run the updated gate launcher, which performs `--preflight_only` before
  starting 2000-step.

## Decision

Do not start 2000-step training until PAI preflight passes.

Starting upstream VideoPainter training here would only train the official
VideoPainter objective, not the requested Exp11 outer b0.75 S2 style DPO
adapter. That would create a mislabeled experiment.

## Required Next Implementation

Before the gate can run on PAI, sync and preflight the isolated trainer:

```text
exp14_adapter_videopainter/code/train_videopainter_dpo_adapter.py
```

It must:

- load a trainable VideoPainter policy;
- load a frozen VideoPainter reference from the same checkpoint;
- convert/read GT winner + generated loser + mask pairs;
- run policy/reference winner/loser forward passes on the same timestep/noise;
- compute region-local MSE with `boundary_mode=outer`,
  `mask=1.0`, `boundary=0.75`, `outside=0.05`;
- compute log-ratio normalized gaps and clipped loser gap;
- write `dpo_diagnostics.csv`;
- save checkpoints and last weights;
- run DAVIS eval with the project metric wrapper.

The trainer is now implemented locally, but DAVIS eval wiring after gate2000 is
still pending and should remain blocked until training itself completes.
