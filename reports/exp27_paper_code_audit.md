# Exp27 Official Paper-Code Audit

## Cached Assets

| Paper | PDF SHA256 | Repo | Commit | License status |
|---|---|---|---|---|
| LocalDPO | `76e8d745153cd08dfc284e116117841c28860de80176497bdbb41f2cc41a0040` | `https://github.com/1170300714/Local-DPO` | `7528e966b17283cfa638577827e456737335f030` | no top-level LICENSE found |
| Diffusion-SDPO | `8137455aa1b655a619eed29bf4ae58b198f04b9e8c0ff35c04a6fd6e3fa4d126` | `https://github.com/AIDC-AI/Diffusion-SDPO` | `84fb241c1b89705a247da8b0d6047798ca49830d` | Apache-2.0 + NOTICE |
| Linear-DPO | `de0c58ea20dc693dd6eb901e016af47c20f8b06699803a04d18af964f87576a7` | `https://github.com/Whynot0101/Linear-DPO` | `663179c7adbbbd2d77b97b5841534447eb291ebd` | no top-level LICENSE content found |

## LocalDPO

Key files:

- `innerT2V/utils/random_mask_gen.py`
- `innerT2V/generate_corrupted_videos.py`
- `innerT2V/generate_corrupted_videos_wan22.py`
- `innerT2V/dataset/t2v_dataset_mask.py`
- `innerT2V/train_cogx.py`
- `innerT2V/train_wanx21.py`

Code evidence:

- `train_cogx.py` loads `pos_videos`, `neg_videos`, `masks`, and `yitas`.
- `train_cogx.py` computes full DPO and mask DPO terms.
- `train_cogx.py` normalizes masked losses with `masks_shape / torch.sum(masks)` and scales by `yitas`.
- `generate_corrupted_videos*.py` generates corrupted videos and masks before training.

Reproduction status:

- Loss/code audit complete.
- Official random-mask execution is currently blocked by a reshape runtime error in `random_mask_gen.py`.

## Diffusion-SDPO

Key files:

- `train.py`
- `scripts/train/sd15_diffusion_dpo.sh`

Code evidence:

- `get_adaptive_lose_l_scale` computes output-space gradient geometry.
- Training uses a detach trick to scale the loser branch gradient.
- Launch exposes `--use_winner_preserving` and `--winner_preserving_mu`.

Reproduction status:

- CPU safe-lambda helper parity passed exactly.
- Full batch gradient-cosine parity is pending.

## Linear-DPO

Key files:

- `train/train_sd_dpo.py`
- `train/train_sd3_dpo.py`
- `utils/train_utils.py`
- `run_sd1_5_pickapic_linear.sh`

Code evidence:

- `train_sd_dpo.py` exposes `--linear_dpo`, `--use_ema_ref`, `--eta_dpo`, and `--decay_ema`.
- Linear utility is computed under `torch.no_grad()` and clamped to `[eta, 1-eta]`.
- EMA reference update occurs after optimizer step.
- SD3 path supports flow-matching target.

Reproduction status:

- CPU utility and EMA update parity passed.
- Full DiffuEraser/VideoPainter save-resume parity is pending.

## Actionable Consequence

Exp27 should not claim novelty over the core mechanisms of these three papers. The primary method decision must be an inpainting-specific preference-data and region/failure-structure decision, with SDPO and Linear-DPO retained as exact baselines or ablations.
