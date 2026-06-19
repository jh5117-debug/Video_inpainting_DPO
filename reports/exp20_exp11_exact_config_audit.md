# Exp20 Exp11 Exact Config Audit

## Source Of Truth

The Exp20 legacy control must reproduce the current best Exp11 Stage1 setting before any image-space boundary search is allowed.

Reviewed sources:

- `exp11_region_boundary_ablation/config.yaml`
- `experiment_registry/exp11_region_boundary_ablation/config.yaml`
- `experiment_registry/exp11_region_boundary_ablation/paths.yaml`
- `scripts/launch_exp11_exp12_parallel_pai.sh`
- `exp11_region_boundary_ablation/code/train_stage1.py`

## Best Exp11 Variant

- name: `exp11_boundary_outer_b075_o005_s1s2_2000`
- boundary mode: `outer`
- mask weight: `1.0`
- boundary weight: `0.75`
- outside weight: `0.05`
- best Stage1 output:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260613_042729_exp11_boundary_exp11_boundary_outer_b075_o005_s1s2_2000_s1_2000_davis_pai`
- best Stage2 output:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2/20260613_042729_exp11_boundary_exp11_boundary_outer_b075_o005_s1s2_2000_s2_2000_davis_pai`

## Data And Weights

- source repaired manifest:
  `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4/manifests/selected_primary_comp.repaired.pai_paths.jsonl`
- GT-win training manifest:
  `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/exp09_10_11_youtubevos_gtwin_d3comp_pai/manifests/selected_primary_comp.gtwin.pai_paths.jsonl`
- SFT-48000 base/ref:
  `/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000`
- train mask mode: `partial`
- mask source: manifest
- task: partial-mask BR/video inpainting
- prior mode: ProPainter as in DiffuEraser data/eval path

## DPO Objective

Confirmed from trainer and launcher:

- `gap_normalization = log_ratio`
- `gap_eps = 1e-6`
- `beta_dpo = 10`
- `lose_gap_weight = 0.25`
- `lose_gap_clip_tau = 1.0`
- `sft_reg_weight = 0.0`
- `winner_abs_reg_weight = 0.05`
- `winner_gap_reg_weight = 1.0`
- `winner_gap_reg_margin = 0.0`
- `winner_gap_reg_mode = relu`

Exp11 loss form:

```text
g_w = log((m_w + eps) / (m_w_ref + eps))
g_l = log((m_l + eps) / (m_l_ref + eps))
g_l_clip = clamp(g_l, max=1.0)
L_DPO = mean[-logsigmoid(-0.5 * 10 * (g_w - 0.25 * g_l_clip))]
L_total = L_DPO + 0.05 * m_w + ReLU(g_w)
```

## Region Map

Confirmed exact legacy behavior:

- DiffuEraser mask convention in trainer: `brushnet_masks`, where `0 = unknown/hole`, `1 = known/context`.
- DPO hole mask: `hole = 1 - brushnet_masks`.
- Latent hard mask: `(hole > 0.5)`.
- Boundary dilation: 3x3 max-pool at latent resolution.
- Exp11 outer boundary: `dilate(mask) - mask`.
- Outside: `1 - clamp(mask_core + boundary, 0, 1)`.
- Weight map values:
  - mask: `1.0`
  - outer boundary: `0.75`
  - outside: `0.05`

Exp20 `legacy_exact=True` and `radius_mode=legacy_latent_exact` must keep this path unchanged.

## Training Schedule And Optimizer

Confirmed from `scripts/launch_exp11_exp12_parallel_pai.sh` and Stage1 defaults:

- GPUs: `0,1,2,3`
- world size: `4`
- per-device batch: `1`
- gradient accumulation: `1`
- effective global batch: `4`
- frames per clip: `16`
- max Stage1 steps: `2000`
- checkpoint steps: `500`
- checkpoint total limit: `5`
- validation steps during training: `999999` (effectively disabled)
- mixed precision: `bf16`
- VAE dtype: `fp32`
- policy dtype: `auto`
- reference dtype: `bf16`
- text dtype: `bf16`
- split positive/negative forward: `true`
- optimizer: `AdamW`
- learning rate: `1e-6`
- scheduler: `constant`
- lr warmup steps: `500` argument remains set but constant scheduler is used
- weight decay: `1e-2`
- gradient clip: `1.0`
- Adam beta1/beta2: `0.9 / 0.999`
- Adam epsilon: `1e-8`

## Exp20 Implication

For first-wave Stage1 search:

- The primary comparison is against Exp11 Stage1 with fixed SFT Stage2, not Exp11 final Stage2.
- `P0 legacy_latent_exact b=0.75` must reproduce the Exp11 Stage1 training/eval behavior before image-space radius trials can be trusted.
- `legacy_exact` parity remains a hard gate.
