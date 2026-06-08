# Exp9: Log-Ratio / Normalized-Gap DPO

Experiment name:

```text
exp9_logratio_gap_dpo_s1s2_2000_davis_pai
```

Goal: solve raw win-gap / lose-gap scale mismatch without deleting the loser
branch. Exp9 remains pairwise DPO.

Loss:

```text
raw_win_gap  = m_w - m_w_ref
raw_lose_gap = m_l - m_l_ref

norm_win_gap  = log((m_w + eps) / (m_w_ref + eps))
norm_lose_gap = log((m_l + eps) / (m_l_ref + eps))
norm_lose_gap_clipped = clamp(norm_lose_gap, max=lose_gap_clip_tau)

inside = -0.5 * beta_dpo * (norm_win_gap - lose_gap_weight * norm_lose_gap_clipped)
L_DPO = mean[-logsigmoid(inside)]
L_total = L_DPO + 0.05 * m_w + 1.0 * ReLU(norm_win_gap - margin)
```

Parameters:

```text
GAP_NORMALIZATION=log_ratio
GAP_EPS=1e-6
LOSE_GAP_CLIP_TAU=1.0
BETA_DPO=10
LOSE_GAP_WEIGHT=0.25
SFT_REG_WEIGHT=0.0
WINNER_ABS_REG_WEIGHT=0.05
WINNER_GAP_REG_WEIGHT=1.0
WINNER_GAP_REG_MARGIN=0.0
```

Run policy:

- Stage1 2000 steps.
- New PAI runs use `NFRAMES=24`.
- DAVIS validation: DPO-S1 + SFT-S2.
- DAVIS validation uses `DAVIS_VIDEO_LENGTH=24`; 16-frame validation is invalid
  because DiffuEraser/ProPainter requires effective duration greater than 22.
- Stage2 2000 steps.
- DAVIS validation: DPO-S1 + DPO-S2.
- Default PAI launch runs Exp9 only.

No-lose-gap is diagnostic only and is not the main method.
