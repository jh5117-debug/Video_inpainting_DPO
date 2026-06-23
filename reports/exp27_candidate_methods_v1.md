# Exp27 Candidate Methods V1

## C1: Task-Native Failure-Structured Preference Data

Problem: Current preference pairs often improve DPO diagnostics without reliably improving video inpainting. We need losers that expose real inpainting failures rather than arbitrary local corruptions.

Formula/data mechanism: construct preference pairs from the actual task condition. BR uses GT winner, masked condition, and self/provided inpainting loser. OR uses `condition=V_obj`, `winner=V_bg`, object mask, raw loser, and affected/unaffected regions. Each loser receives a defect vector.

Difference from LocalDPO: LocalDPO creates random local corruptions in generic video generation. C1 uses task masks and task-native generator failures for inpainting/object removal.

Difference from SDPO: C1 changes data construction, not winner-preserving gradient geometry.

Difference from Linear-DPO: C1 does not change utility shape or reference update.

BR applicability: high. OR applicability: high. Diffusion and flow-matching applicability: data-level, backend-agnostic.

Cost: extra loser generation and defect scoring. No new network.

Required ablations: task-native loser vs LocalDPO-style corruption, single-model vs multi-model loser, medium-hard frontier vs random.

Failure mode: could become dataset engineering without method novelty unless paired with inpainting-specific region constraints.

Paper story: preference quality for restoration depends on failure structure, not only on local masks.

## C2: Restoration-Critical Region Preference Objective

Problem: A single mask-local loss cannot distinguish hole fidelity, seam blending, and observed-context preservation.

Formula: compute separate region losses for BR: mask core, outer seam, outside context; and OR: object mask, affected region, unaffected background. DPO gaps are computed per region and combined with fixed or learned/reportable alphas.

Difference from LocalDPO: LocalDPO has one local corruption mask; C2 decomposes restoration-critical subregions induced by the task.

Difference from SDPO: C2 is region semantics and region aggregation, not safe loser-gradient scaling.

Difference from Linear-DPO: C2 keeps the utility fixed and changes residual geometry.

BR applicability: high. OR applicability: high if affected-region maps are valid.

Cost: low to moderate; no new network.

Required ablations: mask-only, boundary-only, outside preservation, region-balanced vs global weighted mean.

Failure mode: too close to "weighted MSE" unless the subregion diagnostics and BR/OR unification are strong.

Paper story: restoration preference is a structured regional tradeoff, not a binary local-vs-global mask.

## C3: Region-Conditioned Heuristic SDPO

Problem: Winner degradation and loser-dominant margins remain in region-local DPO.

Formula: compute SDPO-style loser scaling from output-space gradients, optionally per task-region. Use it only as a heuristic unless a new theorem is derived for log-ratio/clipped LoVI.

Difference from LocalDPO: adds winner-preserving geometry; LocalDPO has no safe-lambda mechanism.

Difference from SDPO: applies region-conditioned losses and task masks, but guarantee is not inherited without proof.

Difference from Linear-DPO: affects loser branch gradient rather than utility saturation.

Cost: extra autograd and memory. No new network.

Required ablations: exact SDPO, LoVI+SDPO heuristic, per-region vs global lambda.

Failure mode: expensive, unstable in DDP, and not a main novelty if only SDPO plugged into masks.

Paper story: local restoration DPO needs safeguards, but this is likely a support component rather than the primary contribution.

## C4: Stage-Aware Spatial/Temporal Preference Decomposition

Problem: Stage1 and Stage2 optimize different failure types; reusing the same preference data/loss can wash out temporal gains.

Formula/data mechanism: Stage1 receives spatial hole/seam defect pairs; Stage2 receives temporal/flicker/motion-boundary defect pairs and diagnostics. The objective can remain LoVI initially.

Difference from LocalDPO: LocalDPO does not distinguish inpainting stage semantics.

Difference from SDPO: stage-specific data/objective design, not gradient safety.

Difference from Linear-DPO: not a utility/reference update.

Cost: moderate; needs separate manifests and eval gates.

Required ablations: same data both stages vs stage-specific data, Stage1-only vs Stage2-only.

Failure mode: expensive and hard to isolate from data scale or checkpoint selection.

Paper story: video restoration preference has spatial and temporal phases that need different supervision signals.

## C5: Saturation-Aware Utility As Diagnostic, Not Main Method

Problem: sigmoid DPO saturates early, especially with easy losers.

Formula: compare sigmoid DPO, Linear-DPO frozen, Linear-DPO EMA, and LoVI clipping on identical pairs.

Difference from Linear-DPO: none if used directly; therefore not a primary candidate.

Recommended role: baseline/diagnostic for C1/C2, not core novelty.
