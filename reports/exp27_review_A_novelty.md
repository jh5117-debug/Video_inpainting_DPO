# Exp27 Reviewer A Report: Literature / Novelty

Reviewer: A, literature and novelty  
Scope: Exp27 paper-grounded preference study  
Date: 2026-06-23  

I did not read any other Exp27 review reports while preparing this assessment. Evidence below comes from the three extracted paper texts in `/home/hj/video_dpo_paper_code_cache/pdfs/` and the official code cache in `/home/hj/video_dpo_paper_code_cache/repos/`, plus Exp27/lineage project materials needed to understand the claimed method surface.

## Executive Assessment

The novelty position for Exp27 is viable only if it is narrowed to video inpainting-specific preference optimization: known context, task masks, hole/boundary/context weighting, winner anchoring, normalized or clipped gap control, and evaluation under inpainting metrics. Broad claims about "localized DPO for video diffusion", "region-aware video DPO", "real/GT winner versus corrupted/generated loser pair construction", or "DPO stability for diffusion models" are no longer safe.

The highest-risk overlap is LocalDPO. It independently claims real-video positives, locally corrupted model-generated negatives, mask-guided local preference signals, and a region-aware DPO objective for video diffusion. Diffusion-SDPO threatens any broad claim about winner preservation or safe DPO updates. Linear-DPO threatens broad loss-objective novelty claims around sigmoid DPO failure, sustained/non-saturating utilities, EMA references, and unified diffusion/flow preference optimization.

The safest paper framing is therefore:

- Not "first localized DPO for video diffusion."
- Not "first region-aware/mask-aware DPO."
- Instead: "boundary-aware, region-local preference optimization for conditional video inpainting/object removal, where the known context and task mask make the preference problem different from text-to-video local corruption."

## Novelty Threat Summary

| Prior work | Threat level | Main overlap | What remains distinguishable |
|---|---:|---|---|
| LocalDPO, arXiv 2601.04068 | High | Real/clean positive videos, locally corrupted negative videos, mask-normalized local DPO, hybrid DPO/SFT objective, critique of global video DPO | Exp27 is conditional video inpainting/object removal, uses task masks and known context, emphasizes boundary/context preservation and inpainting metrics |
| Diffusion-SDPO, arXiv 2511.03317 | Medium | DPO can enlarge preference gap while harming winner; proposes winner-preserving safe loser gradient scaling | Exp27 uses region-local weighting and explicit winner-anchor/gap regularizers rather than gradient-geometry safe-lambda; not the same algorithm |
| Linear-DPO, arXiv 2605.21123 | Medium | Sigmoid DPO pseudo-convergence, clipped/sustained linear utility, EMA reference, diffusion/flow-matching generality | Exp27's contribution should be task/regional inpainting formulation, not a general DPO objective for generative models |

## Evidence From LocalDPO

LocalDPO is the central novelty threat.

### Paper Overlap

LocalDPO's abstract states that it constructs localized preference pairs from real videos, treats high-quality real videos as positives, corrupts random spatio-temporal regions with a frozen base model to form negatives, and applies a region-aware DPO loss restricted to corrupted areas (`2601.04068.txt`, Abstract, lines 20-55). This directly overlaps the generic version of an Exp27 claim such as "use a ground-truth or real winner and generated loser, with a mask-local DPO loss."

The introduction makes the same motivation Exp27 has used for video inpainting DPO: existing video DPO methods rely on multiple generated samples or reward/critic ranking, operate on global quality scores, and ignore local artifacts (`2601.04068.txt`, Sec. 1, lines 84-123). The paper's contributions explicitly include real-video positives, locally corrupted negatives, avoiding expensive multi-sample ranking/annotations, and a mask-guided region-aware DPO loss (`2601.04068.txt`, lines 130-145).

The technical overlap is concrete:

- Standard video DPO is formalized in latent space in Eq. (1)-(2), using winner/loser reconstruction-error differences against a frozen reference (`2601.04068.txt`, Sec. 3, lines 150-188).
- The limitation of standard video DPO is described as global, monolithic preference learning that ignores localized cues (`2601.04068.txt`, Sec. 3.2, lines 215-237).
- LocalDPO constructs preferred/dispreferred pairs by taking high-quality real videos as preferred and using localized corruption as dispreferred, with no human or reward labels (`2601.04068.txt`, Sec. 4.1, lines 239-254).
- Eq. (3) performs masked local corruption in latent denoising: the masked region comes from the generated/corrupted latent, and the outside region remains from the original latent (`2601.04068.txt`, Sec. 4.2, lines 255-288).
- Eq. (4)-(5) define a region-aware preference loss and mask-normalized error difference (`2601.04068.txt`, Sec. 4.3, lines 300-321 and following Eq. (5) excerpt).
- Eq. (6) uses a hybrid objective combining region-aware DPO, standard full-latent DPO, and SFT regularization (`2601.04068.txt`, Sec. 4.3, Eq. (6), excerpt lines 264-285).

LocalDPO also already acknowledges limitations that are close to likely Reviewer 2 concerns for Exp27: random Bezier masks are not semantic and future work could use GroundingDINO/SAM (`2601.04068.txt`, limitations, lines 648-671). If Exp27 uses task masks from the inpainting/object-removal setting, that is a real differentiator, but only if stated as inpainting-specific and not as generic local DPO novelty.

### Code Overlap

The official LocalDPO repository operationalizes the same ingredients:

- `repos/Local-DPO/README.md`, lines 81-131, documents custom training data with `pos_video_path`, `neg_video_path`, `mask`, `yita`, and `gen_caption`, plus generation of corrupted videos before training.
- `repos/Local-DPO/innerT2V/dataset/t2v_dataset_mask.py`, lines 291-354, loads positive video, negative video, mask, and corruption/noise severity `yita`, checks pair consistency, resizes masks to latent resolution, and returns them to training.
- `repos/Local-DPO/innerT2V/generate_corrupted_videos.py`, lines 407-499, creates original videos, edited/corrupted videos, and masks; lines 420-435 sample random connected components; lines 462-477 call the video pipeline with input frames, mask, and corruption severity.
- `repos/Local-DPO/innerT2V/train_cogx.py`, lines 187-195, loads a frozen reference transformer; lines 595-603 read pos/neg/mask/yita; lines 680-736 compute global and masked DPO terms, reference terms, SFT terms, and total loss.
- `repos/Local-DPO/innerT2V/train_wanx21.py`, lines 574-700, implements the same training structure for Wan2.1.

This code means a reviewer can reasonably say LocalDPO is not merely conceptual prior art; it is an implemented video-diffusion local DPO baseline.

### Consequence For Exp27 Claims

The following claim forms are not safe:

- "First localized DPO for video diffusion."
- "First region-aware/mask-guided video DPO."
- "First to use real/GT videos as winners and corrupted/generated videos as losers without human labels or reward models."
- "First to show global video DPO misses local artifacts."
- "First hybrid local DPO plus SFT approach."

The safe distinction is conditional video inpainting. LocalDPO is text-to-video generation using synthetic random local corruption. Exp27 can still claim a task-specific formulation where the inpainting mask, known unmasked context, hole boundary, and temporal object-removal setting define the preference region and evaluation target. That distinction must be explicit and repeated.

## Evidence From Diffusion-SDPO

Diffusion-SDPO is a novelty threat to any broad "winner preservation" or "DPO stability" claim.

### Paper Overlap

The abstract states that standard Diffusion-DPO can increase reconstruction error for both winner and loser even as the preference margin improves, and proposes SDPO to preserve the winner by adaptively scaling the loser gradient according to winner/loser gradient alignment (`2511.03317.txt`, Abstract, lines 20-46). The introduction frames the method as a safe-lambda mechanism that modifies DPO objectives and is plug-compatible with Diffusion-DPO, DSPO, and DMPO (`2511.03317.txt`, lines 70-109).

The paper's technical claim is directly relevant to Exp27's DPO diagnostics:

- Eq. (5)-(9) restate diffusion DPO as a log-ratio objective over winner/loser residual comparisons (`2511.03317.txt`, Sec. 3, lines 146-188).
- Sec. 3.2 says standard DPO gives no guarantee that winner loss decreases; over-penalizing the loser can worsen the preferred sample (`2511.03317.txt`, lines 196-212).
- Sec. 4 derives a safety condition so preferred loss does not increase after a first-order update (`2511.03317.txt`, lines 214-260).
- Eq. (15)-(17) and Algorithm 1 compute an output-space proxy safe scale for the loser branch (`2511.03317.txt`, lines 264-333).
- The limitations section concedes the guarantee is first-order and can fail under curvature or noisy/bias coefficient estimates (`2511.03317.txt`, lines 520-537 and appendix lines 675-752).

### Code Overlap

The official code implements the winner-preserving scale:

- `repos/Diffusion-SDPO/README.md`, lines 11-12, describes a plug-in rule that computes an adaptive loser-branch scale from winner/loser output-space gradients.
- `repos/Diffusion-SDPO/README.md`, lines 48-51, exposes `--use_winner_preserving` and `--winner_preserving_mu`.
- `repos/Diffusion-SDPO/train.py`, lines 84-124, implements `get_adaptive_lose_l_scale`, computing output-space gradients for winner and loser MSE terms, the dot product, the norm ratio, and a clamped scale.
- `repos/Diffusion-SDPO/train.py`, lines 1176-1238, applies the scale to the loser loss via a detach trick before computing DPO/DSPO/DMPO losses.

### Consequence For Exp27 Claims

Exp27 should not claim general novelty for preventing DPO winner degradation. It can claim a different practical design if supported by ablations: explicit winner absolute regularization, winner-gap regularization, clipped loser gap, and normalized gaps in the video inpainting setting. But the paper must cite SDPO and explain why it did not simply use safe-lambda or include it as a baseline.

A strong reviewer criticism will be: "Your diagnosis of raw DPO instability is already known; SDPO proposes a direct safeguard. Why is your winner anchoring better or more appropriate for inpainting?"

## Evidence From Linear-DPO

Linear-DPO is a novelty threat to broad claims about fixing DPO's sigmoid objective, gradient saturation, clipping, sustained utility, or EMA references.

### Paper Overlap

The abstract says existing DPO studies are mainly for denoising diffusion, overlook flow matching, and suffer objective mismatch; the paper derives generalized DPO for diffusion and flow matching through a reverse-time SDE and proposes Linear-DPO, replacing sigmoid utility with sustained linear utility and using an EMA-updated reference (`2605.21123.txt`, Abstract, lines 12-41).

The paper directly covers several loss-level ideas that Exp27 must not overclaim:

- Sec. 4.1 derives a unified DPO objective for diffusion and flow matching, Eq. (7), (8), and (11) (`2605.21123.txt`, lines 164-199).
- Sec. 4.2 interprets DPO as weighted SFT and argues standard sigmoid weighting is ill-suited to regression-based diffusion objectives, producing pseudo-convergence or requiring very small learning rates (`2605.21123.txt`, lines 201-248).
- Eq. (15)-(16) define Linear-DPO with a clipped linear utility, stop-gradient weighting, and EMA reference (`2605.21123.txt`, lines 254-325).
- Sec. 4.3 explicitly removes the external beta scaling and uses bounded linear weights so the method can reuse SFT learning rates (`2605.21123.txt`, lines 330-351).
- Experiments cover SD1.5, SDXL, and SD3-M flow matching (`2605.21123.txt`, lines 389-481), with ablations on utility choice and beta sensitivity (`2605.21123.txt`, lines 538-603 and 1371-1396).

### Code Overlap

The official code implements clipped linear weighting and optional EMA reference:

- `repos/Linear-DPO/train/train_sd_dpo.py`, lines 357-363, exposes `--linear_dpo`, `--eta_dpo`, `--use_ema_ref`, `--decay_ema`, and `--valid_ema`.
- `repos/Linear-DPO/train/train_sd_dpo.py`, lines 1132-1170, computes model/reference winner-loser MSE differences; with `linear_dpo`, it forms `0.2 * beta_dpo * (model_diff - ref_diff) + 0.5`, clamps it, and uses it as a stop-gradient weight on `model_losses_w - model_losses_l`.
- `repos/Linear-DPO/train/train_sd_dpo.py`, lines 1196-1199, updates EMA reference parameters after optimizer steps.
- `repos/Linear-DPO/train/train_sd3_dpo.py`, lines 400-419, exposes the same SD3/flow flags; lines 1180-1221 implement the flow-matching DPO and linear-DPO losses; lines 1240-1247 update the EMA reference.
- `repos/Linear-DPO/utils/train_utils.py`, lines 114-155, implements `ModelEMA`.

One reproducibility caveat: the Linear-DPO scripts are not perfectly uniform. For example, `run_sd1_5_pickapic_linear.sh` passes both `--linear_dpo` and EMA flags, while `run_sdxl_pickapic_linear.sh` passes linear-DPO flags without EMA, and `run_sd3_hpdv3_linear.sh` appears not to pass `--linear_dpo` or EMA flags despite its name. This does not remove the paper's novelty threat, but it should make any baseline reproduction careful and explicitly configured.

### Consequence For Exp27 Claims

Exp27 should not claim broad novelty for "fixing DPO's sigmoid/logistic saturation", "using clipped linearized preference weights", "EMA reference DPO", or "general DPO for diffusion/flow models." If Exp27 uses normalized log-ratio gaps or clipped loser terms, it should frame them as inpainting-stabilization design choices and cite Linear-DPO as related objective-level work.

## Exp27 Claims That No Longer Hold

The following claim families should be removed or narrowed:

1. "We introduce the first local/region-aware DPO for video diffusion."
   LocalDPO already claims and implements region-aware local DPO for video diffusion, including mask-normalized losses and local corrupted negatives.

2. "We are first to build preference pairs from clean real videos and locally corrupted/generated videos without human labels."
   LocalDPO uses real videos as preferred samples and model-corrupted local regions as dispreferred samples.

3. "Existing video DPO methods only optimize global preference and ignore local artifacts."
   This remains true for many older methods, but it is no longer a novelty claim after LocalDPO. It can be used as background only.

4. "Mask-guided DPO is the core novel method."
   Mask-guided DPO is already central to LocalDPO. Exp27's novelty must be in inpainting-specific masks, boundary treatment, context preservation, and inpainting evaluation.

5. "Our winner anchor is the first DPO safeguard for winner preservation."
   Diffusion-SDPO directly targets winner preservation by scaling loser gradients.

6. "Our clipped/normalized preference loss is the first solution to DPO objective mismatch or saturation."
   Linear-DPO directly argues that sigmoid DPO is mismatched to diffusion/flow regression and proposes clipped sustained linear weights.

7. "The method is broadly applicable to flow-matching video models as a new DPO derivation."
   Linear-DPO already derives unified diffusion/flow DPO, and Diffusion-SDPO includes a FLUX flow-DiT extension.

8. "Flow-prior or semantic-mask claims" if the implementation does not actually use them.
   Existing project notes already warn not to claim real optical-flow prior consistency for older proxy experiments. LocalDPO also calls semantic mask placement future work, so unsupported semantic claims will be easy to attack.

## Claims Still Safe With Proper Framing

The following claim families remain defensible if backed by ablations and carefully worded:

1. Conditional video inpainting preference optimization.
   Exp27 can claim a task-specific formulation for video inpainting/object removal where the input video, binary task mask, and known context constrain the optimization target. This is different from LocalDPO's text-to-video local corruption setup.

2. Boundary-aware region decomposition.
   The hole/boundary/context decomposition is a credible inpainting-specific contribution. Project code in `dataset/region_mask_utils.py`, lines 1-91, defines mutually exclusive hole, boundary, and context regions from the binary inpainting mask. This is not the same as LocalDPO's random Bezier local mask.

3. Context preservation in known unmasked regions.
   In video inpainting, the unmasked context is not just background in a generated sample; it is an observed conditioning signal that should remain stable. Claims around reducing background/context drift are safer than broad claims about generic video quality preference.

4. Winner anchoring and clipped loser/gap control as an inpainting stabilization package.
   The project materials describe a progression from raw/global DPO to GT winner pairs, normalized log-ratio gap, clipped loser gap, region-local loss, boundary-aware weighting, and winner-anchor regularization (`PRD/25_paper_materials_and_writing_plan.md`, lines 19-38 and 126-144). This can be positioned as a practical inpainting recipe, not a general DPO theory contribution.

5. Empirical finding that full-frame/raw DPO is misaligned for video inpainting.
   This remains publishable if shown with diagnostics, metrics, and visual failures, but the text must cite LocalDPO and SDPO as related diagnoses in generation settings.

6. Evaluation on video inpainting metrics and object-removal benchmarks.
   LocalDPO reports text-to-video generation quality and user studies; Exp27 can distinguish itself by DAVIS/YouTubeVOS-style inpainting metrics, hole/boundary/context breakdowns, temporal consistency, and final-only VOR-Eval if kept out of method selection.

7. Integration into the existing inpainting training stack.
   Wrappers such as `DPO_finetune/train_dpo_stage1.py` and `DPO_finetune/dataset/dpo_dataset.py`, plus the region mask utilities, point to an inpainting-specific engineering context. This is a weaker novelty claim by itself, but useful for reproducibility.

## Reviewer 2 Criticisms To Anticipate

1. "This is essentially LocalDPO for video inpainting."
   The answer must be: LocalDPO is the closest prior; our contribution is not generic local DPO but boundary-aware, context-preserving, conditional video inpainting DPO. Include a baseline or ablation that implements the closest LocalDPO-style objective in the inpainting setting.

2. "The paper overclaims firstness."
   Remove all first/novel claims about local masks, region-aware DPO, real-positive/corrupted-negative pairs, DPO winner preservation, sigmoid-DPO saturation fixes, and diffusion/flow DPO generality.

3. "Why not compare to LocalDPO's RA-DPO objective?"
   At minimum include an ablation: mask-only LocalDPO-style RA-DPO versus Exp27 hole/boundary/context weighting plus winner anchoring and clipped gap control. If not possible, state it as a limitation, but this is a serious weakness.

4. "Why not use Diffusion-SDPO?"
   Include SDPO as related work and either compare to safe-lambda scaling or explain why gradient-geometry loser scaling is orthogonal to inpainting region weighting. A combined variant would be a strong rebuttal experiment if feasible.

5. "Your objective-level modifications overlap Linear-DPO."
   Cite Linear-DPO in the objective/loss section. Avoid presenting clipping or sustained weights as original in isolation. If using linear or clipped weights, compare to a Linear-DPO-style variant under the same inpainting data.

6. "The gains are small and may be selection artifacts."
   Existing project materials cite gains such as DAVIS50 PSNR 32.7314 to 33.0140 and YouTubeVOS100 33.3968 to 33.7238 for an Exp11-style stage-2 setting (`PRD/25_paper_materials_and_writing_plan.md`, lines 76-77). These are plausible but modest. Report confidence intervals, paired tests, or per-sequence win rates where possible.

7. "VOR-Eval or any learned evaluator influenced method choice."
   The PRD says VOR-Eval should be final-only and not used for selection. Maintain that separation.

8. "The method still optimizes the loser more than the winner."
   Project notes acknowledge loser-dominant diagnostics even after stabilization (`PRD/25_paper_materials_and_writing_plan.md`, lines 126-128). Report these diagnostics honestly and frame the method as improved stability, not a formal winner-preserving guarantee.

9. "Semantic masks or flow priors are claimed but not implemented."
   Do not claim real flow-prior consistency or semantic mask placement unless implemented and ablated. Project notes already flag old flow-proxy claims as unsafe.

10. "Official prior code already supports your setup."
   LocalDPO's code has explicit positive/negative/mask/yita metadata and masked DPO training. The paper must explain why task masks, known context, and boundary weighting create a materially different problem.

## Recommended Related-Work Positioning

Use LocalDPO as the closest local-preference prior:

"LocalDPO localizes preference learning for text-to-video diffusion by constructing real-positive and locally corrupted-negative pairs and restricting the DPO signal to random spatio-temporal corruptions. Our setting differs in that video inpainting supplies an observed source video and task mask; the unmasked context and mask boundary are conditioning constraints, not merely unpreferred generated regions. We therefore decompose the inpainting mask into hole, boundary, and context terms and stabilize DPO around a ground-truth winner under inpainting metrics."

Use Diffusion-SDPO as the winner-preservation prior:

"SDPO addresses winner degradation by adaptively scaling loser gradients using first-order gradient geometry. Our regularizers are complementary and task-local: they constrain winner reconstruction and gap behavior under region-local inpainting losses. We do not claim a formal monotonic winner-loss guarantee."

Use Linear-DPO as the objective-mismatch prior:

"Linear-DPO shows that sigmoid DPO can pseudo-converge for diffusion/flow regression losses and proposes clipped linear weighting with an EMA reference. Our normalized/clipped gap terms are used for inpainting stability and are evaluated in the conditional video inpainting setting rather than presented as a universal DPO objective."

## Minimum Experiments Or Ablations Needed For A Defensible Novelty Story

1. LocalDPO-style mask-only baseline in the inpainting setting.
   Use GT/real winner, generated/corrupted loser, and mask-normalized DPO over the hole only, without boundary/context decomposition or winner anchoring. This directly addresses the strongest novelty threat.

2. Boundary decomposition ablation.
   Compare hole-only, hole+boundary, hole+boundary+context, and the chosen boundary weight. The safest novelty claim depends on boundary/context effects.

3. Winner-anchor and clipped-gap ablation.
   Show raw DPO, normalized gap, clipped loser gap, winner-anchor regularization, and the full objective. Include winner/loser loss diagnostics, not only final image/video metrics.

4. SDPO-style safeguard comparison or hybrid.
   Even a small comparison would reduce reviewer pressure. If a full baseline is too costly, report why safe-lambda was not adopted and leave it as future work.

5. Linear-DPO-style loss comparison.
   Compare standard logistic DPO versus a clipped linear-weight DPO variant under the same region-local inpainting data.

6. No-selector final evaluation.
   Keep VOR-Eval final-only and report conventional inpainting metrics plus visual failure cases.

## Bottom-Line Novelty Verdict

As currently framed, Exp27 has a high risk of overclaiming because LocalDPO already covers the broad idea of localized region-aware DPO for video diffusion with real positives and locally corrupted negatives. Diffusion-SDPO and Linear-DPO further occupy the general objective-stability space.

The work can still be novel enough if the paper is rewritten around a narrower and more defensible contribution: video inpainting-specific preference optimization with known context, task masks, boundary-aware region weighting, and empirically stabilized winner/loser gap behavior. The title, abstract, introduction, and contributions should make this restriction obvious. The strongest version of the paper treats LocalDPO as the closest prior and demonstrates that inpainting needs more than LocalDPO's random-mask RA-DPO: it needs boundary/context-aware losses and winner-stabilized training under the conditional video inpainting objective.
