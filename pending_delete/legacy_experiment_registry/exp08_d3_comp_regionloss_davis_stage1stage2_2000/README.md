# Exp8 D3 Comp Region-Loss DAVIS S1/S2 2000

Exp8 is a target-domain diagnostic, not a long-training direction. It uses D3 YouTube-VOS comp generated-loser data for training and validates on DAVIS partial-mask inpainting.

The experiment tests whether region-weighted MSE inside the regularized DPO objective improves mask and boundary behavior when partial-mask inpainting is evaluated with ProPainter prior and the 48000-step SFT DiffuEraser weights.

## Training Chain

- Stage1: 2000 steps, D3 comp manifest, partial mask from manifest, region-weighted DPO.
- Stage1 validation: build `DPO-S1 + frozen SFT-S2` hybrid weights, then run DAVIS validation.
- Stage2: 2000 steps, initialized from Stage1, same DPO params and region loss.
- Stage2 validation: run `DPO-S1 + DPO-S2` DAVIS validation.

## Loss

Notation:

- `m_w`: policy winner region-weighted MSE
- `m_l`: policy loser region-weighted MSE
- `m_w_ref`: reference winner region-weighted MSE
- `m_l_ref`: reference loser region-weighted MSE
- `win_gap = m_w - m_w_ref`
- `lose_gap = m_l - m_l_ref`

For this experiment:

```text
beta_dpo = 10
lose_gap_weight = 0.25
sft_reg_weight = 0.0
winner_abs_reg_weight = 0.05
winner_gap_reg_weight = 1.0
winner_gap_reg_margin = 0.0

L_total =
    -logσ{-0.5 * 10 * (win_gap - 0.25 * lose_gap)}
    + 0.05 * m_w
    + ReLU(win_gap)
```

Here `m_w`, `m_l`, `m_w_ref`, and `m_l_ref` are computed with region weights:

```text
mask region     = 1.0
boundary region = 0.5
outside region  = 0.05
```

## Validation

Validation must use partial-mask video inpainting metrics and four-column visualizations:

1. winner / GT
2. mask overlay
3. DiffuEraser-base, using SFT-48000
4. current experiment

VBench is not used for this inpainting task.
