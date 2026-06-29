# Exp43 H20 MiniMax Stage2 SFT Runner

H20-only isolated MiniMax Stage2 SFT runner track.

Current status: `H20_EXP43_BF16_SAFE_READY`.

PAI remains read-only. New code is limited to
`exp43_h20_minimax_stage2_sft_runner/`.

Readback confirms the Exp41 blocker was runner scope, not missing H20 data or
weights: previous MiniMax runners cap at 10 steps, while Exp43 is authorized to
add an isolated true 30/100/300-step Stage2 SFT ladder runner.

BF16-safe preflight P0-P7 passed on H20, including DDP8 one-batch training and
rank0 checkpoint reload. This is runtime-only evidence, not a MiniMax quality
claim.
