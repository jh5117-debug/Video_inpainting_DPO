# Exp52 R1 Row0 Smoke

Status: `VOID_R1_ROW0_SMOKE_PASS`

## Setup

- Recipe: R1 WinnerPreserve-LocalDPO
- Quadmask: `q0_current`
- Timestep: `500`
- Scope: `proj_out`
- Cache file: `/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp52_void_winner_preserving_allgpu/cache/tensor_cache/q0_current/train4/00_BLENDER_CON001_00636.pt`
- Optimizer: AdamW, lr=1e-05, steps=1

## Checks

- Runtime: 632.97 sec
- Checkpoint: `/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp52_void_winner_preserving_allgpu/r1_row0_smoke/checkpoints/r1_q0_t500_proj_out_row0_step1.pt`
- Strict reload: True
- Policy param delta > 0: True (max norm 0.005100787617266178)
- Reference delta: 0.0
- Winner loss finite: True
- Loser loss finite: True
- Grad finite: True
- Winner gap post: 0.0001377984881401062
- Loser gap post: 1.9371509552001953e-06
- Effective loser contribution ratio: 0.014258294488620784
- Forward after reload finite: True
- Peak reserved VRAM: 20.053 GiB

## Interpretation

R1 row0 produced a checkpoint inside the bounded runtime using cached VOID-native inputs. The loser branch has `loser_grad_scale=0.0`, so loser degradation is not an active gradient driver in this smoke.
