# VideoPainter DPO Trainer Preflight

Date: 2026-06-16 CST
Host: dsw-753014-dc85766cb-4v2jj

## Status

blocked_before_preflight

## Reason

The required VideoPainter/CogVideoX weights are unavailable and PAI cannot access Hugging Face.

## What Passed

- Clean Exp14 worktree exists.
- Exp14 trainer `py_compile` passed.
- Exp14 gate launcher `bash -n` passed.
- VideoPainter code repo was synced to PAI.
- YouTube-VOS train data exists.
- DAVIS eval data exists.
- Generated-loser manifest exists and does not contain `/home/nvme01`.
- GPUs are available.
- HF token environment variables are set.

## Blocker

Missing:

```text
third_party/VideoPainter/ckpt/CogVideoX-5b-I2V
third_party/VideoPainter/ckpt/VideoPainter/checkpoints/branch
```

Attempted HF download:

```text
hf download TencentARC/VideoPainter
```

Failed with:

```text
httpx.ConnectError: [Errno 101] Network is unreachable
```

## Decision

Do not run trainer preflight. Without the weights, the trainer cannot construct the policy/reference VideoPainter branches and cannot compute `m_w`, `m_l`, `m_w_ref`, or `m_l_ref`.
