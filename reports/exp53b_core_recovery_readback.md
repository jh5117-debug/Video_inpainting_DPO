# Exp53B Core Recovery Readback

Status: `EXP53B_READY_FOR_CORE_CELLS`

Timestamp: `2026-07-01T15:20:19+00:00`
Branch: `research/exp53-void-r1r2-targeted-h20-20260701`
HEAD: `9f9d99f9a998671edec519bc4506a6ac57146f54`

## Scope

Exp53B resumes the Exp53 H20 lane only for the two Q2/T500 core cells:

- `R1_Q2_T500_S0`
- `R2_Q2_T500_S0`

No T300, R3/R4, 10-step, VOR-Eval, hard comp, universal adapter, final SOTA, or third-backbone claim is allowed.

## Git Readback

`git fetch --all --prune` was attempted on H20 but timed out/hung on GitHub network and was interrupted. The local branch is already at the user-provided Exp53 HEAD `9f9d99f9a`; previous pushes verified remote/local HEAD.

## GPU0-3

| GPU | Used MiB | Total MiB | Util % | Status |
| --- | ---: | ---: | ---: | --- |
| 0 | 28 | 97871 | 0 | `free_xorg_only` |
| 1 | 1 | 97871 | 0 | `free` |
| 2 | 1 | 97871 | 0 | `free` |
| 3 | 1 | 97871 | 0 | `free` |


Free GPUs usable for Exp53B: `0,1,2,3`

Compute apps snapshot:

```text
/home/ubuntu/.profile: line 29: /home/nvme02/GR00T/GR00T-WholeBodyControl/.tools/uv/env: No such file or directory
/home/ubuntu/.profile: line 31: /home/nvme02/GR00T/PRM/.tools/uv/env: No such file or directory
```

Project process snapshot:

```text
/home/ubuntu/.profile: line 29: /home/nvme02/GR00T/GR00T-WholeBodyControl/.tools/uv/env: No such file or directory
/home/ubuntu/.profile: line 31: /home/nvme02/GR00T/PRM/.tools/uv/env: No such file or directory
ubuntu    389559  389554  389559  389559       05:41 Ss   bash -c hostname; date -Ins; cd /home/nvme01/H20_Video_inpainting_DPO_exp53_void_r1r2_h20 && git fetch --all --prune || true && git branch --show-current && git rev-parse HEAD && git status --short && git log -8 --oneline && git diff --stat && git diff --check
ubuntu    791937  789329  791937  791937       00:08 Ss   bash -c cd /home/nvme01/H20_Video_inpainting_DPO_exp53_void_r1r2_h20 && python3 /tmp/exp53b_audit.py && git diff --check && git status --short && git diff --stat
ubuntu    792007  791937  791937  791937       00:08 Sl   python3 /tmp/exp53b_audit.py
```

## Q2/T500 Cache Audit

Cache root: `/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp52_void_winner_preserving_allgpu/cache/tensor_cache/q2_strict_affected`

- train4 files: 4
- heldout4 files: 4
- required tensors present: `True`
- fixed timestep: `500`
- same noise/timestep: checked per row

Exp52 cache parity remains `VOID_CACHE_PARITY_EXPLAINED`; the scalar reference-loss bf16 downcast caveat is known and does not require cache regeneration for Q2/T500.

## Decision

Proceed directly to Milestone C if status is `EXP53B_READY_FOR_CORE_CELLS`. Milestone B cache repair is skipped unless the cache becomes inconsistent.
