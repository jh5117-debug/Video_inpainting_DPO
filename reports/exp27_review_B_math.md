# Exp27 Reviewer B: Mathematical Correctness Review

Scope: I read the extracted paper texts for LocalDPO (`2601.04068.txt`), Diffusion-SDPO (`2511.03317.txt`), and Linear-DPO (`2605.21123.txt`), inspected their official cached repos under `/home/hj/video_dpo_paper_code_cache/repos`, and inspected the current Exp10/Exp11-style loss code in this repo. I did not read other Exp27 review reports.

## Executive Verdict

The strongest mathematically defensible claim is the SDPO claim: under a simplified preference objective and infinitesimal gradient descent, choosing a loser-gradient scale below a geometry-derived bound makes the winner loss non-increasing to first order. Even this is explicitly first-order and approximate in the official paper, because the practical implementation uses output-space gradients, a slack parameter, clipping, stochastic minibatches, and finite optimizer steps.

The LoVI / local video inpainting loss used in this repo is best described as a principled heuristic inspired by LocalDPO plus stabilizers from SDPO/Linear-DPO. Region weighting, log-ratio gaps, loser clipping, and winner anchors can each be justified mathematically as gradient shaping or soft regularization, but none by themselves prove monotone winner improvement, human-preference improvement, or global video quality preservation.

## Notation

For a preferred/winner sample `w` and rejected/loser sample `l`, let

`m_w(theta) = ||y_w - f_theta(z_w,t,c)||^2`, `m_l(theta) = ||y_l - f_theta(z_l,t,c)||^2`,

and similarly `r_w`, `r_l` for the frozen reference. The standard diffusion-DPO residual gaps are

`g_w = m_w - r_w`, `g_l = m_l - r_l`.

The canonical objective is a logistic preference loss of the form

`L_DPO = - log sigma( - beta (g_w - g_l) / 2 )`

up to paper-specific timestep weights and sign conventions. This is exactly the structure stated in LocalDPO Eq. 1-2 (`2601.04068.txt:174-186`), SDPO Eq. 8-9 (`2511.03317.txt:161-184`), and Linear-DPO Eq. 6 / Eq. 11 (`2605.21123.txt:143-153`, `2605.21123.txt:185-193`).

## LocalDPO / LoVI Region Loss

LocalDPO replaces full-latent residual gaps by masked residual gaps. The paper defines a localized tuple `(c, x_w, x_l, M, alpha)`, where `M` marks corrupted regions (`2601.04068.txt:230-237`), and a region-aware gap

`Delta'_*(theta) = (N_M / ||M||_1) [ || M o (y_* - f_theta(z_*,t,c)) ||^2 - || M o (y_* - f_ref(z_*,t,c)) ||^2 ]`

in Eq. 5 (`2601.04068.txt:255-260`). Its local DPO objective is

`L_RA-DPO = - E log sigma( - beta (1 + eta(alpha)) E_t[Delta'_w - Delta'_l] )`

from Eq. 4 (`2601.04068.txt:300-320`), and the paper combines it with global DPO and SFT:

`L_total = lambda_RA L_RA-DPO + lambda_DPO L_DPO + lambda_SFT L_SFT`

from Eq. 6 (`2601.04068.txt:264-284`).

The official Local-DPO code matches the mathematical structure: in `Local-DPO/innerT2V/train_cogx.py`, it computes full and masked policy residuals, masked reference residuals, `model_diff_mask - ref_diff_mask`, and `loss_dpo_mask = -F.logsigmoid(inside_term_mask)` (`train_cogx.py:680-736`). The mask is also used during negative generation/fusion in `pipeline_cogvideox_improved_dense_dpo_mask.py:354-362` and `pipeline_cogvideox_improved_dense_dpo_mask.py:428-467`.

What can be proven:

- Region weighting changes the optimized norm. If `W >= 0`, the per-sample weighted MSE
  `m_W = sum_i W_i e_i^2 / sum_i W_i`
  has gradient `d m_W / d pred_i = 2 W_i (pred_i - y_i) / sum_j W_j`. Thus larger weights provably allocate larger local gradient magnitude per residual.
- The normalization by `sum W` makes the loss invariant to multiplying all region weights by a common constant.
- If the corrupted region is the only systematic winner/loser difference, then focusing `W` on that region reduces irrelevant background variance in the empirical loss.

What is heuristic:

- The preference order "real video beats locally corrupted video" is a data-construction assumption, not a theorem. It is plausible when the real clip is high quality and the corruption is visible, but it can fail for noisy real videos, semantically irrelevant masks, or cases where the generated local patch looks better.
- Local region improvement does not imply global video improvement unless the global and local objectives are aligned. The paper handles this empirically with the hybrid objective, not with a formal guarantee.
- The LocalDPO paper does not prove winner preservation. It does not bound `Delta m_w` under gradient descent.

## Current LoVI / Exp11-Style Loss In This Repo

The current local implementation in `training/dpo/train_stage1.py` is more conservative than paper LocalDPO. It uses a region-weighted MSE, log-ratio gaps, loser-gap clipping, a reduced loser weight, and winner-anchor penalties.

Region weighting is implemented as

`m_W = sum(err * weight) / clamp(sum(weight), 1e-8)`

at `training/dpo/train_stage1.py:360-375`. The region map uses the DiffuEraser convention `hole = 1 - mask`, gives mask, boundary, and outside weights, and returns stats at `training/dpo/train_stage1.py:576-610`. Exp10/Exp11 set `mask=1.0`, `boundary=0.5`, `outside=0.05` (`exp10_region_local_dpo/config.yaml:14-17`, `exp11_flow_prior_consistency_dpo/config.yaml:17-20`). This is not a pure masked loss: outside pixels still receive 5% relative weight.

The gap transform is

`raw_win = m_w - r_w`, `raw_lose = m_l - r_l`,

`norm_win = log((m_w + eps)/(r_w + eps))`, `norm_lose = log((m_l + eps)/(r_l + eps))`,

then either raw or log-ratio gaps are used (`training/dpo/train_stage1.py:408-425`). Exp10/Exp11 use `gap_normalization: log_ratio` and `gap_eps: 1e-6` (`exp10_region_local_dpo/config.yaml:5-6`, `exp11_flow_prior_consistency_dpo/config.yaml:8-9`).

The implemented DPO inside term is

`s = -0.5 beta [ g_w - alpha_l g_l^clip ]`,

`L_dpo = mean(-log sigma(s))`,

where `g_l^clip = clamp(g_l, max=tau)` (`training/dpo/train_stage1.py:427-435`). Exp10/Exp11 set `beta=10`, `alpha_l=0.25`, `tau=1.0` (`exp10_region_local_dpo/config.yaml:7-9`, `exp11_flow_prior_consistency_dpo/config.yaml:10-12`).

The winner anchor is

`L_anchor = lambda_abs m_w + lambda_gap mean([g_w - margin]_+)`,

added to the DPO loss (`training/dpo/train_stage1.py:436-450`). Exp10/Exp11 use `lambda_abs=0.05`, `lambda_gap=1.0`, `margin=0` (`exp10_region_local_dpo/config.yaml:11-13`, `exp11_flow_prior_consistency_dpo/config.yaml:14-16`).

Mathematical effects:

- Region weighting: provably changes gradient allocation toward the mask and boundary. It does not prove better inpainting, because that depends on the correctness of the region labels and on model capacity.
- Log-ratio: makes gaps relative to the reference scale. `log(m/r)` has the same sign as `m-r`, but derivative `1/(m+eps)` amplifies small-MSE examples and suppresses high-MSE examples. This can improve conditioning, but it is not the DPO likelihood derived from a Gaussian transition model.
- Loser clipping: caps positive loser degradation contribution at `tau`, reducing the incentive to win by making losers arbitrarily worse. Because it is a `max` clamp only, very negative loser gaps remain unbounded in the other direction. This is a useful anti-collapse heuristic, not a preservation proof.
- Reduced loser weight: `alpha_l=0.25` moves the objective toward winner improvement and away from loser degradation. It still lacks SDPO's per-step gradient-geometry condition.
- Winner anchor: penalizes absolute winner error and policy-worse-than-reference winner gaps. It gives a soft regularization pressure. It does not ensure `m_w(theta_{k+1}) <= m_w(theta_k)` for a finite optimizer step.

## SDPO Guarantee

SDPO is the only reviewed method with a formal preservation claim. The paper analyzes

`L_pref(theta) = L_w(theta) - lambda L_l(theta)`

and a gradient step

`Delta theta = - eta (grad L_w - lambda grad L_l)`.

The first-order winner change is

`Delta L_w ~= grad L_w^T Delta theta = -eta ||grad L_w||^2 + eta lambda grad L_w^T grad L_l`.

The paper's printed Eq. 12 omits the plus sign visually because it writes the update convention compactly, but the safety inequality and bound are clear: to make `Delta L_w <= 0`, require

`lambda <= ||grad L_w||^2 / (grad L_w^T grad L_l)`

when `grad L_w^T grad L_l > 0`; if the dot product is non-positive, the loser branch is intrinsically safe (`2511.03317.txt:196-236`).

The practical paper approximation replaces parameter-space gradients by output-space gradients:

`lambda_safe = (1 - mu) ||g_w||^2 / (g_w^T g_l)`,

with clipping to `[0,1]` and `lambda_safe=1` when `g_w^T g_l <= 0` (`2511.03317.txt:290-330`). The official code implements this in `Diffusion-SDPO/train.py:84-124`, then scales only the loser branch by the detach trick

`model_losses_l_scaled = detach(model_losses_l) + lambda_safe * (model_losses_l - detach(model_losses_l))`

before applying the normal logistic DPO loss (`Diffusion-SDPO/train.py:1187-1209`).

Limits of the guarantee:

- It is first-order only. The paper's appendix states that the true change contains `1/2 Delta theta^T H_w Delta theta + O(||Delta theta||^3)`, so finite steps and curvature can still increase winner loss (`2511.03317.txt:675-695`).
- The output-space version is a proxy for the parameter-space bound; the Jacobian geometry factor is absorbed into slack `mu`.
- Adam, gradient clipping, distributed minibatches, mixed precision, and changing timesteps all move the real update away from the clean derivation.
- SDPO's theorem applies to the specific loss whose gradients define `L_w` and `L_l`. To inherit it in LoVI, one would need to compute `lambda_safe` using the same region-weighted/log-ratio winner and loser residuals, and then scale only the loser gradient. The current winner anchor and loser clipping do not replicate that proof.

## Linear-DPO Utility

Linear-DPO first derives a unified DPO objective for diffusion and flow-matching as a logistic loss over policy/reference squared-error differences (`2605.21123.txt:172-199`). Its gradient can be interpreted as weighted SFT descent on the winner and ascent on the loser:

`grad L(theta) = E[ beta_bar sigma(beta_bar Delta D_theta) grad( ||y_w-y_theta||^2 - ||y_l-y_theta||^2 ) ]`

from Eq. 13 (`2605.21123.txt:203-231`). The paper argues that the sigmoid weight saturates too quickly for regression-style image generation, then proposes a clipped linear utility:

`omega'(Delta D) = clip(0.2 beta_bar Delta D + 0.5, eta, 1)`

and loss

`L_Linear-DPO = E[ stopgrad(omega') ( ||y_w-y_theta||^2 - ||y_l-y_theta||^2 ) ]`

from Eq. 15-16 (`2605.21123.txt:306-327`, `2605.21123.txt:329-350`).

The official code implements the stop-gradient utility through `torch.no_grad()` and then multiplies it by `(model_losses_w - model_losses_l)` in both SD/SDXL and SD3 trainers (`Linear-DPO/train/train_sd_dpo.py:1162-1166`, `Linear-DPO/train/train_sd3_dpo.py:1214-1218`). The reference model is used to compute the utility ratio, not directly inside the final residual-difference term. EMA reference support is implemented by wrapping the reference model in `ModelEMA` (`Linear-DPO/train/train_sd_dpo.py:616-617`, `Linear-DPO/train/train_sd3_dpo.py:662-663`) and updating it during training (`Linear-DPO/train/train_sd_dpo.py:1198-1199`, `Linear-DPO/train/train_sd3_dpo.py:1244-1246`). One minor code/paper mismatch: the paper text says clip to `[eta, 1]`, while the official code clamps to `[eta, 1-eta]`.

What can be proven:

- Within the unclipped range, the linear utility avoids sigmoid saturation and supplies a constant-slope weighting function.
- The clipped utility bounds the scalar multiplier, so the preference part cannot apply arbitrarily large per-pair scalar weights.
- Stop-gradient makes the implemented gradient exactly the chosen scalar weight times the SFT-difference gradient, rather than including second-order utility-gradient terms.

What is heuristic:

- Linear utility is not derived as the Bradley-Terry likelihood optimum. It intentionally changes the objective for optimization behavior.
- It gives no winner-preservation guarantee. If loser ascent conflicts with winner descent, Linear-DPO can still raise winner error.
- EMA reference is a stabilization heuristic. It changes the moving baseline and can help continued learning, but it weakens any fixed-reference likelihood interpretation.

## Recommendations For Exp27 Claims

Use strong language only for these points:

- "Region weighting provably reallocates residual-gradient mass toward mask/boundary pixels under the implemented weighted MSE."
- "Log-ratio gaps are scale-normalized policy/reference comparisons whose sign agrees with raw gaps for positive MSEs."
- "Loser clipping and reduced loser weight cap/suppress one route to loser-dominant wins."
- "Winner-anchor terms directly penalize winner degradation relative to absolute MSE and to the reference."
- "SDPO provides a first-order, infinitesimal-step winner non-increase guarantee under its gradient-geometry assumptions."

Avoid claiming:

- That LoVI/Exp11 proves monotonic winner improvement.
- That region weighting proves human-preference improvement.
- That log-ratio DPO remains the exact likelihood derived from diffusion transition probabilities.
- That loser clipping is equivalent to SDPO.
- That Linear-DPO is theoretically safer than sigmoid DPO. It is better conditioned by design, not winner-preserving by proof.

Best mathematically precise positioning:

`LoVI` should be framed as a paper-grounded composite objective:

`L_LoVI = -E log sigma(-0.5 beta [ phi_W(m_w,r_w) - alpha_l clip_tau(phi_W(m_l,r_l)) ])`

`+ lambda_abs m_w^W + lambda_gap [phi_W(m_w,r_w) - delta]_+`,

where `phi_W` is either raw `m_W-r_W` or log-ratio `log((m_W+eps)/(r_W+eps))`. This objective inherits LocalDPO's localized residual focus, Linear-DPO-like conditioning intuition through transformed/bounded scalar weights, and SDPO's motivation to protect winners, but not SDPO's theorem unless a true geometry-based loser-gradient scale is added.
