# Exp14 Status

Status: **isolated trainer implemented; PAI blocked before preflight because VideoPainter weights are missing and HF is unreachable**.

What passed:

- Local VideoPainter repo found.
- Official training entrypoints found.
- VideoPainter is CogVideoX / DiT diffusion-based.
- Training loop exposes timestep, noise, target, mask, model prediction, and
  checkpoint saving.
- A frozen reference model is conceptually possible by loading a second copy of
  the same pretrained branch / transformer under `torch.no_grad()`.

What changed on 2026-06-15:

- Added isolated trainer:
  `exp14_adapter_videopainter/code/train_videopainter_dpo_adapter.py`
- The trainer reads the current frame-directory DPO manifest with
  `win_video_path`, `final_loser_video_path`, and `mask_path`.
- It defines policy/reference VideoPainter branch forwards on shared
  timestep/noise and computes `m_w`, `m_l`, `m_w_ref`, `m_l_ref`.
- It implements Exp11 outer b0.75 S2 style region-local normalized-gap
  clipped-loser-gap winner-anchored DPO.
- It writes `dpo_diagnostics.csv` and supports `--preflight_only`.

What has not passed yet:

- PAI did not run the required trainer preflight because the hard weight
  precheck failed.
- The 2000-step gate has not been launched.
- Multi-GPU sharding is not implemented in the isolated trainer; the first
  PAI preflight must verify memory feasibility.

Latest PAI attempt:

```text
date = 2026-06-16 CST
sync_strategy = clean_worktree
clean_repo = /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate
status = blocked_before_preflight
blocker = missing CogVideoX-5b-I2V base model and VideoPainter branch checkpoint;
          hf download fails with Network is unreachable
```

What passed on PAI:

- Exp14 clean worktree synced to commit `2e187ee`.
- Trainer `py_compile` passed.
- Gate launcher `bash -n` passed.
- VideoPainter code repo was rsynced from HAL.
- YouTube-VOS, DAVIS, and DPO manifest exist.
- Manifest does not contain `/home/nvme01`.
- GPUs are available.

Missing weights:

```text
third_party/VideoPainter/ckpt/CogVideoX-5b-I2V
third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch
```

HF attempt:

```text
hf download TencentARC/VideoPainter
-> httpx.ConnectError: [Errno 101] Network is unreachable
```

Decision:

```text
Do not launch 2000-step training from the upstream VideoPainter script alone.
Provide/mount the VideoPainter weights first, then rerun the gate2000 script;
it runs `--preflight_only` first and launches gate2000 only if
policy/reference DPO loss and backward pass succeed.
```
