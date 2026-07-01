# Exp57 Qualitative Summary

H20 Exp57 generated heldout4 Step0/Step1 videos and evidence sheets for all four adaptive cells.

Codex inspected the all-sample overview evidence sheets for:

- `ATS0_Q2_T500_S0`
- `ATS_STRICT_Q2_T500_S0`
- `ATS_HALFLR_Q2_T500_S0`
- `ATS_NODPO_Q2_T500_S0`

No cell reached visual PASS. `ATS_STRICT_Q2_T500_S0` was least bad but remained `1 better / 0 tie / 3 worse`. `ATS_HALFLR` and `ATS_NODPO` were `0 better / 0 tie / 4 worse`.

Readback qualitative pattern: Exp55 and Exp56 are mixed-only. Object/mask can improve and outside remains safe, but transition regions remain visually and metrically fragile. Exp56-H20 R5 and R5_HALF both had 0 better / 2 tie / 2 worse visual outcomes.

Exp57 H20 strengthens the diagnosis: even with adaptive lambda and no-loser-DPO diagnostic, overlap / affected / boundary regions remain fragile. No third-backbone evidence is claimed.
