# Exp49 ROSE Readback

Status: `EXP49_PAI_ACCESS_BLOCKED_BEFORE_DOWNLOAD`

This report performs the Milestone A readback and public source audit only. It does not claim PAI execution.

## Environment Readback

Current shell:

```text
hostname = hal-9000
cwd = /home/hj
PAI NAS path = missing
PAI workspace path = missing
```

Git note: `git fetch --all --prune` was attempted from the local mirror but did
not return within the working window and was interrupted. The Exp49 worktree was
created from the locally available `origin/main` at
`34844d75aba585542b311098417f67c7274f6434`.

PAI access probes:

| probe | result |
| --- | --- |
| `/mnt/nas/hj/H20_Video_inpainting_DPO` on local shell | missing |
| `/mnt/workspace/hj/nas_hj` on local shell | missing |
| `dsw-753014-85f54df947-bkp7h` DNS | unresolved from HAL |
| `47.103.26.60:22` | TCP open |
| `root/hj/ubuntu@47.103.26.60` with `codex_pai` or `hj_pai_ed25519` | SSH command timed out before returning hostname |

Conclusion: this session cannot verify that it is running on PAI. No asset download, environment install, inference, GPU use, or training-forward audit was started.

## Current Positive Backbone Evidence

DiffuEraser remains the main BR positive line under its own DAVIS50 protocol. The current paper-facing table in prior reports lists Exp11 outer b0.75 S2 as the main DiffuEraser BR evidence, with DAVIS50 raw6 hard-comp metrics:

```text
PSNR = 32.840213
SSIM = 0.971818
LPIPS = 0.015339
Ewarp = 7.181782
mask PSNR = 21.196763
boundary PSNR = 26.441316
```

VideoPainter is the second positive adapter backbone. Exp31 reports `VIDEOPAINTER_2000_POSITIVE` with fixed search-dev and shadow-dev evidence. Key shadow-dev Step2000 versus Step0 gains:

```text
full PSNR +6.2632
mask PSNR +10.8860
sampled boundary PSNR +12.2343
win rate 1.0000
```

Exp31 also reports completed LPIPS and mask-region Ewarp review and visual inspection of all search-dev and shadow-dev evidence pages.

## Why MiniMax Is Not Enough As Third Adapter

MiniMax has substantial plumbing evidence, but it remains negative as adapter evidence:

- Exp35 found MiniMax utility scale too weak and output movement too small to matter.
- Exp36 concluded MiniMax is plumbing-positive, trainability-positive, and inference-sensitivity-positive, but not quality-positive.
- Exp37 LocalDPO/bad-noise recipes produced nonzero movement but no heldout visual positive gate.
- Exp47 forensic audit found pseudo-success SFT should be localized, not global, and MiniMax remains not third-backbone evidence.

Therefore MiniMax should not be promoted as the third adapter. It is an audited limitation / unresolved candidate.

## Why ROSE Is A Better Next Candidate

ROSE is explicitly designed for removing objects with side effects in videos. Public project material and HF metadata identify it as a video-to-video diffusion-transformer inpainting model based on `alibaba-pai/Wan2.1-Fun-1.3B-InP`. Its task language and dataset tags include object removal, video inpainting, shadow removal, and reflection removal, which align more directly with VOR-OR than MiniMax's generic removal behavior.

Expected useful properties to audit on PAI:

- full-video input conditioning;
- mask / side-effect handling;
- Wan-style transformer training path;
- difference-mask predictor in the HF Space file inventory;
- potential trainable `WanTransformer3D` module.

## Public ROSE Assets

| asset | public location | metadata readback |
| --- | --- | --- |
| Project page | `https://rose2025-inpaint.github.io/` | public project page |
| Code | `https://github.com/Kunbyte-AI/ROSE` | `main` HEAD `6be41c5420bf331c6d491277d5a6feaf9b3a779a` |
| HF model | `https://huggingface.co/Kunbyte/ROSE` | public, not gated, SHA `8a5e57c9f25e73a4b62d324719cdf3367c13df59` |
| HF Space | `https://huggingface.co/spaces/Kunbyte/ROSE` | public, not gated, SHA `0ea1fc65605d8734bd85df2c12d8198687cc4229` |
| HF dataset | `https://huggingface.co/datasets/Kunbyte/ROSE-Dataset` | public, not gated, SHA `94159b52c62c914dbf86ab6be4cbae6039cae9aa` |

## Licenses

| asset | license |
| --- | --- |
| Code | Apache-2.0 |
| HF model | Apache-2.0 |
| HF dataset | CC-BY-NC-4.0 |
| HF Space | no explicit license in API card data; must inspect repository files during asset download |

## Download Without Token

HF API metadata reports the model, dataset, and space as public and not gated. Therefore the first PAI attempt should use direct unauthenticated download. If a large-file backend, quota, or mirror issue appears, record the exact failure and only then use H20 relay as a transfer staging path.

## Potential Auth Or Network Issues

No token should be printed or stored in reports. Possible blockers:

- HuggingFace large file download timeout;
- `hf_xet` availability;
- mirror lag if using `HF_ENDPOINT`;
- dataset size/quota;
- base model `alibaba-pai/Wan2.1-Fun-1.3B-InP` may have additional large files and should be audited separately on PAI.

## PAI Direct Failure Plan

1. Try PAI direct download first.
2. If PAI network/HF fails, use H20 only as a download relay, with no H20 GPU and no H20 training.
3. Generate sha256 on H20 staging.
4. Rsync to PAI target.
5. Verify sha256 on PAI.
6. Delete H20 staging only after verification, if requested.

## Exact Promotion Gate

ROSE can be promoted only stepwise:

```text
ROSE_ASSETS_READY
ROSE_ENV_READY
ROSE_TRUE_ADAPTER_FEASIBLE
ROSE_INFERENCE_SMOKE_PASS or technically valid weak smoke
ROSE_VOR_OR_GATE16_PASS or technically valid weak gate
ROSE_ONE_STEP_PASS
ROSE_ADAPTER_10STEP_PROMISING or ROSE_ADAPTER_10STEP_POSITIVE
```

No inference-only result may be described as adapter positive.

## Milestone A Answer Checklist

1. Current positive backbone evidence: DiffuEraser and VideoPainter.
2. MiniMax is not enough as third adapter: quality-positive gates failed despite plumbing and trainability.
3. ROSE is a better next candidate: task and architecture align with object-plus-side-effect removal.
4. Public ROSE assets exist: project page, GitHub code, HF model, HF dataset, HF Space.
5. Code / Demo / Dataset / Model availability: public metadata indicates available and not gated.
6. Licenses: Apache-2.0 for code/model, CC-BY-NC-4.0 for dataset, Space license to inspect after clone.
7. Download without token: likely code/model/dataset/space metadata; actual large files must be tested on PAI.
8. May require auth: no gated flag observed, but base model and large-file access must be verified.
9. If PAI direct download fails: H20 relay with sha256 verification, no H20 GPU/training.
10. Exact promotion gate: listed above.

## Blocker

Milestone B is blocked until a real PAI shell or reachable PAI SSH endpoint is available.
