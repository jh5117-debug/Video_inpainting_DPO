# VideoPainter Adapter Gate2000 Precheck

Date: 2026-06-15

## Status

Blocked. The 2000-step gate was **not** launched.

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

Failed:

- The isolated adapter trainer is missing:
  `exp14_adapter_videopainter/code/train_videopainter_dpo_adapter.py`
- Therefore the gate cannot compute:
  - `m_w`
  - `m_l`
  - `m_w_ref`
  - `m_l_ref`
  - region-local normalized-gap DPO loss
  - VideoPainter adapter diagnostics

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

Failed:

- PAI checked repos did not contain `exp14_adapter_videopainter`.
- PAI checked repos did not contain
  `exp14_adapter_videopainter/code/train_videopainter_dpo_adapter.py`.
- The VideoPainter repo was not found in the checked project paths.

## Decision

Do not start 2000-step training.

Starting upstream VideoPainter training here would only train the official
VideoPainter objective, not the requested Exp11 outer b0.75 S2 style DPO
adapter. That would create a mislabeled experiment.

## Required Next Implementation

Before the gate can run, implement an isolated trainer:

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

