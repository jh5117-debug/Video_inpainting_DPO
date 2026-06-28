# Exp40 MiniMax PSNR-Safe Rescue Registry

Exp40 tests whether the small Exp38 R1 raw-PSNR signal can be converted into a
boundary/outside-safe MiniMax adapter improvement using PSNR-safe SFT and only
then a restricted DPO-after-SFT gate.

Current result: `MINIMAX_SFT_PSNRSAFE_NEGATIVE`. The 12-recipe 30-step SFT
grid failed on search; do not run 100-step, DPO-after-SFT, or 300/500-step
confirmation from the current recipes.
