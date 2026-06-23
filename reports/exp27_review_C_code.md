# Exp27 Reviewer C Implementation Review

Scope: implementation reproducibility for Exp27. I reviewed the supplied extracted paper texts in `/home/hj/video_dpo_paper_code_cache/pdfs/{2601.04068,2511.03317,2605.21123}.txt`, the official code cache under `/home/hj/video_dpo_paper_code_cache/repos`, current DiffuEraser DPO trainers, and the available VideoPainter adapter trainer. I did not read other Exp27 review reports.

## Executive Finding

Exp27 is not yet implementable as an exact reproduction. The Exp27 directories and registries are initialized but empty, and there are no Exp27 launch configs, parity tests, loss modules, or official-code mappings populated. The current DiffuEraser trainers contain a useful DPO training skeleton, but their objective is a local project variant: region-weighted/log-ratio DPO with loser clipping and winner regularizers. It is not exact LocalDPO, Diffusion-SDPO, or Linear-DPO.

Minimum viable Exp27 should start with a paper-grounded LocalDPO implementation on top of the DiffuEraser trainer, with explicit loss-mode flags, manifest fields for local corruption metadata, and toy/parity tests before any long training. SDPO is a small but gradient-sensitive loss extension. Linear-DPO is higher risk because it requires an EMA reference model and save/resume semantics that current DiffuEraser code does not have.

## Current Exp27 State

- `/exp27_paper_grounded_preference_study/README.md` and `status.md` are placeholders only.
- `/experiment_registry/exp27_paper_grounded_preference_study/status.md` is initialized only.
- `paper_inventory.csv`, `paper_code_mapping.csv`, and `paper_code_commits.csv` contain headers but no rows.
- `/exp27_paper_grounded_preference_study/code`, `configs`, `scripts`, `parity`, and `runs` contain no implementation artifacts.

This means exact reproducibility currently depends on retrofitting shared trainers rather than running an Exp27-owned implementation.

## Paper and Official Code Evidence

### LocalDPO, `2601.04068`

Best fit for Exp27 video inpainting preference optimization. The paper constructs localized preference pairs `(c, x_w, x_l, M, alpha)` where `x_w` is the original video and `x_l` is a locally corrupted/inpainted negative under mask `M`. The loss combines global video Diffusion-DPO, region-aware DPO, and winner SFT:

- Global DPO uses the model-vs-reference MSE gap for winner and loser.
- Region-aware DPO masks the squared errors and normalizes by mask area.
- The region term is scaled by a function of corruption strength `alpha`/`yita`.
- Paper settings include LoRA rank 64 on DiT attention, lambda weights `1.0/1.0/0.1`, batch size 128, 540 iterations, AdamW, and inference with 50 DDIM steps and CFG 6.0.

Official code touchpoints:

- `Local-DPO/innerT2V/train_cogx.py`: policy/reference loading and frozen reference around lines 180-248; LoRA setup around 253-262.
- `Local-DPO/innerT2V/train_cogx.py`: batch fields `pos_videos`, `neg_videos`, `masks`, `yitas` around 586-604.
- `Local-DPO/innerT2V/train_cogx.py`: shared noise/timestep pair construction around 605-643.
- `Local-DPO/innerT2V/train_cogx.py`: global and masked losses around 680-736.
- `Local-DPO/innerT2V/dataset/t2v_dataset_mask.py`: manifest/dataset loading of positive, negative, mask, and optional `yita` around 288-356.
- `Local-DPO/innerT2V/generate_corrupted_videos_wan22.py`: local corrupted-video and mask generation around 394-490.
- `Local-DPO/innerT2V/commandline/train_base.sh`: launch defaults include bf16, batch 1, gradient accumulation 4, beta 5000, LoRA rank 64, and loss weights. In this cache it points to `innerT2V/train.py`, which is not present, so the script is not directly runnable without selecting the Cog/Wan trainer.

Gap against DiffuEraser:

- DiffuEraser has region-weighting, but not exact LocalDPO mask-area normalization plus global DPO plus region-aware DPO plus SFT as separate paper terms.
- Current generated-loser manifests do not appear to enforce LocalDPO's required `(mask, alpha/yita, corruption-generation checkpoint, mask polarity)` contract.
- Current loss has project-specific log-ratio normalization, loser-gap clipping, `lose_gap_weight`, and winner regularizers. These must be disabled or separated for exact LocalDPO.

### Diffusion-SDPO, `2511.03317`

This is an extension of Diffusion-DPO that preserves winner optimization by adaptively scaling the loser branch gradient. The implementation-critical detail is that the loser loss value remains in the objective, but its gradient is multiplied by a detached scalar:

`L_l_scaled = L_l.detach() + lambda_safe * (L_l - L_l.detach())`

Official code touchpoints:

- `Diffusion-SDPO/train.py`: `get_adaptive_lose_l_scale` around lines 84-124 computes output-space gradients for winner and loser losses, then clamps lambda.
- `Diffusion-SDPO/train.py`: DPO training around 1080-1195 shares noise/timesteps across winner/loser and applies the detach trick around line 1192.
- `Diffusion-SDPO/scripts/train/sd15_diffusion_dpo.sh`: launch uses Accelerate, bf16, lr `1e-8`, max steps 2000, warmup 500, beta 5000, and `--use_winner_preserving --winner_preserving_mu 0.9`.

Gap against DiffuEraser:

- No current SDPO lambda computation exists in `compute_dpo_loss`.
- Current `split_pos_neg_forward` can run the loser policy branch under `torch.no_grad()` when `lose_gap_weight == 0`; that would break SDPO because SDPO requires loser branch gradients before scaling.
- SDPO lambda should be computed in float32 under mixed precision, detached, logged, and guarded for `den <= 0`.

### Linear-DPO, `2605.21123`

Linear-DPO replaces the sigmoid utility with a clipped linear utility and commonly uses an EMA reference model. Official code clips the stop-gradient ratio to `[eta_dpo, 1 - eta_dpo]`, even though the paper text describes the upper clip as `1`.

Official code touchpoints:

- `Linear-DPO/train/train_sd_dpo.py`: args for `--linear_dpo`, `--use_ema_ref`, `--decay_ema`, and `--eta_dpo` around 357-364.
- `Linear-DPO/train/train_sd_dpo.py`: reference initialization around 605-617; static frozen ref or `ModelEMA`.
- `Linear-DPO/train/train_sd_dpo.py`: `accelerator.prepare` prepares policy/optimizer/dataloader/scheduler only, not the EMA reference, around 927.
- `Linear-DPO/train/train_sd_dpo.py`: DPO and Linear-DPO loss around 1148-1170.
- `Linear-DPO/train/train_sd_dpo.py`: EMA reference update after synced optimizer steps around 1196-1200.
- `Linear-DPO/train/utils/train_utils.py`: `ModelEMA` around 114-177.
- `Linear-DPO/train/run_sd1_5_pickapic_linear.sh`: contains an apparent shell bug, `DECAYS=(...)` but loop uses `"${DECAY[@]}"`.

Gap against DiffuEraser:

- Current reference logic is static frozen reference from `--ref_model_path`; there is no EMA reference.
- Current checkpoints save Accelerate state for prepared trainable modules and final `unet_main`/`brushnet` weights. EMA state would not be saved/resumed unless explicitly added.
- DiffuEraser has multiple trainable/frozen components, so an EMA reference must define exactly which modules are tracked for Stage 1 and Stage 2.

## Current DiffuEraser Trainer Assessment

Primary files:

- `training/dpo/train_stage1.py`
- `training/dpo/train_stage2.py`

Loss files/functions:

- Stage 1 owns `compute_dpo_loss` in `training/dpo/train_stage1.py` around 325-573.
- Stage 2 imports the Stage 1 loss/helpers rather than defining an independent objective.
- The loss computes policy/ref MSEs, optional region-weighted MSE, raw or log-ratio gaps, optional loser gap clipping, `-logsigmoid`, optional SFT regularization, and winner absolute/gap regularizers.

This is a good insertion point, but it should not be treated as a paper reproduction. It is a mixed local objective.

Reference update logic:

- Stage 1 loads policy from `--policy_init_path` or `--ref_model_path`, then loads a separate frozen reference from `--ref_model_path`.
- Stage 2 follows the same static-reference pattern.
- No reference updates occur after optimizer steps.
- Therefore current code can reproduce static-reference Diffusion-DPO variants only, not Linear-DPO's EMA-reference setting.

DDP and mixed precision:

- Both stages use Accelerate and prepare only trainable policy modules, optimizer, dataloader, and scheduler.
- Frozen reference modules are moved to device/dtype but are not wrapped by Accelerate. This is acceptable for a static reference, but insufficient for EMA save/resume unless custom handling is added.
- `policy_dtype` is effectively constrained to auto/fp32 while mixed precision controls broader autocast behavior. SDPO's adaptive gradient calculation should force float32 for stability.
- Stage 1/2 checkpointing uses `accelerator.save_state`, while final/best weights save only `unet_main` and `brushnet`. Any EMA ref, SDPO counters, or objective-specific state must be explicitly serialized.

Hook/save risks:

- There are no custom Accelerate save/load hooks for objective-specific reference state.
- `accelerator.load_state` resumes prepared modules and optimizer, but not a non-prepared EMA wrapper.
- Final `last_weights` and validation `best_weights` do not include objective metadata beyond the run config/logs.

## Available VideoPainter Trainer Assessment

I did not find an Exp26 implementation directory in this tree. The available VideoPainter trainer is `exp14_adapter_videopainter/code/train_videopainter_dpo_adapter.py`, which I treated as the current VideoPainter adapter baseline.

Implementation facts:

- It is explicitly isolated under Exp14 and does not modify upstream VideoPainter or shared DiffuEraser DPO code.
- It is a plain PyTorch single-process loop, not Accelerate/DDP.
- It loads a trainable `policy_branch` and frozen `reference_branch`.
- It uses a region-weighted latent loss over mask, boundary, and outside regions.
- Its objective is log-ratio DPO with loser-gap clipping and winner regularizers, not exact Diffusion-DPO, LocalDPO, SDPO, or Linear-DPO.
- Checkpoints save only the policy branch plus optimizer state.

Launch config facts:

- `launch_videopainter_adapter_gate2000_pai.sh` runs a preflight and then 2000 steps with bf16, beta 10, loser weight 0.25, winner absolute reg 0.05, winner gap reg 1.0, mask/boundary/outside weights `1.0/0.75/0.05`.
- `run_videopainter_adapter_smoke1_pai.sh` and `run_videopainter_adapter_smoke20_pai.sh` call unsupported `--smoke`/`--tiny_val_videos` flags, so they are stale against the current Python parser.

Conclusion: the VideoPainter adapter is useful as prior engineering context, but it is not an Exp27 reproducibility baseline.

## Blocking Reproducibility Risks

1. No Exp27-owned implementation or launch configs exist.
2. No objective selector separates exact paper losses from local experimental DPO variants.
3. LocalDPO's manifest contract is missing: mask polarity, mask resolution, corruption strength `alpha/yita`, corruption generator, and positive/negative provenance must be explicit.
4. Linear-DPO's EMA reference cannot be reproduced with current static-reference loading and checkpointing.
5. SDPO can silently become wrong if loser policy predictions are computed under `torch.no_grad()` or if lambda is computed in low precision.
6. Official code launch scripts are not all directly runnable from the cache, so Exp27 must record exact paper-code commit, patch assumptions, and runnable launch translations.
7. Stage 1/2 final weights do not save objective-specific state; this is a blocker for EMA-reference methods.

## Minimum Viable Exp27 Implementation

Recommended first implementation: LocalDPO on DiffuEraser Stage 1, then Stage 2 only after Stage 1 parity passes. This is the closest paper-to-task match and avoids EMA complexity.

Required changes:

1. Add an explicit objective selector, for example `--dpo_objective {diffueraser_current,localdpo,sdpo,linear_dpo}`. Keep current behavior behind `diffueraser_current`.
2. Add or extract a loss module with unit-testable functions for standard Diffusion-DPO, LocalDPO region-aware DPO, SDPO loser scaling, and Linear-DPO stop-gradient utility.
3. Implement LocalDPO exact loss:
   - binary mask `M` with documented polarity,
   - masked error normalization by mask area,
   - corruption-strength scale from `alpha/yita`,
   - separate global DPO, region-aware DPO, and SFT terms,
   - paper weights defaulting to `1.0, 1.0, 0.1`,
   - diagnostics for each term, mask occupancy, and alpha/yita distribution.
4. Extend `generated_loser_manifest` or add a LocalDPO manifest with `winner_path`, `loser_path`, `mask_path`, `prompt`, `alpha/yita`, `mask_polarity`, `mask_resolution`, and loser-generation metadata.
5. Add SDPO only as an optional objective modifier:
   - compute output-space gradients in float32,
   - apply the detached loser-scaling trick,
   - log lambda mean/min/max and `den <= 0` rate,
   - disable any no-grad loser policy shortcut.
6. Add Linear-DPO only after EMA infrastructure exists:
   - initialize EMA reference from the policy after `policy_init_path`,
   - update from `accelerator.unwrap_model(...)` after synced optimizer steps,
   - save/load EMA state in checkpoints and final artifacts,
   - log reference drift and EMA decay.
7. Create Exp27 launch configs:
   - `smoke_1step` with all objective diagnostics enabled,
   - `parity_1batch` with fixed seed/noise/timestep,
   - `localdpo_stage1_minimal` with no long run until parity gates pass.
8. Add tests/parity checks:
   - toy tensor equality for standard DPO and LocalDPO mask normalization,
   - SDPO gradient scaling check showing loser gradient is scaled but value path remains intact,
   - Linear-DPO check that the utility weight is stop-gradient,
   - save/resume test if EMA is enabled.

## Decision

Do not launch Exp27 long training yet. The current codebase can support a minimal, paper-grounded LocalDPO implementation with moderate changes, but exact reproducibility requires objective separation, manifest hardening, launch configs, and parity tests first. SDPO is a reasonable second step. Linear-DPO should wait until EMA reference save/load and DDP semantics are implemented and tested.
