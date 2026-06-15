# Exp11 Region Boundary Ablation

Purpose: isolate the boundary definition in Exp10-style region-local DPO.

This experiment does not reuse or modify old Exp11-proxy code. Training entrypoints are copied into `code/` and invoked with `DPO_STAGE1_ENTRYPOINT` / `DPO_STAGE2_ENTRYPOINT`.

Variants:

- `exp11_boundary_inner_b075_o005_s1s2_2000`: inner mask boundary, boundary weight 0.75.
- `exp11_boundary_outer_b075_o005_s1s2_2000`: outer mask boundary, boundary weight 0.75.
- `exp11_boundary_both_b075_o005_s1s2_2000`: inner + outer boundary, boundary weight 0.75.
- `exp11_boundary_both_b100_o005_s1s2_2000`: inner + outer boundary, boundary weight 1.0.

Fixed setting follows Exp10: GT winner, generated rollout loser, partial-mask inpainting, SFT-48000 reference, ProPainter prior, DAVIS raw6 D+G off validation.
