# Exp27 Reviewer D: Experimental Design Review

## Scope

Reviewer D was asked to assess Exp27 from an experimental-design perspective only. I reviewed the full extracted paper texts for LocalDPO (`2601.04068`), Diffusion-SDPO (`2511.03317`), and Linear-DPO (`2605.21123`), the Exp25/Exp26 status and reports, the Exp27 PRD/status, and the cached official code for Local-DPO, Diffusion-SDPO, and Linear-DPO. I did not read any other Exp27 review reports.

## Verdict

Exp27 should proceed only as a gated design study first, not as an immediate long-training run. The main experimental risk is not whether LocalDPO/SDPO/Linear-DPO are plausible; it is leakage and confounding. VOR-Eval, DAVIS50, and YouTubeVOS100 must remain final-only, method selection must happen on locked search-dev only, and every proposed paper-derived change must be isolated so the team can tell whether gains come from data construction, objective stabilization, loss weighting, larger data scale, or evaluator reuse.

The strongest experimental candidate is a LocalDPO-style data baseline adapted to OR: real/background targets as winners, same-source localized or generator-corrupted losers, explicit masks, and unchanged inference/eval protocol. SDPO and Linear-DPO are better treated as objective-only ablations layered after the data baseline is proven, because their official code and papers focus on image preference pairs and objective dynamics rather than video-inpainting split hygiene.

## Required Split Contract

Exp27 must create a single immutable split manifest before any loser generation or training:

- `train_pool`: source clips used for training preference pairs.
- `search_dev`: fixed checkpoint/method-selection set.
- `shadow_dev`: untouched until a short-list method passes search-dev.
- `final`: VOR-Eval 43, DAVIS50, YouTubeVOS100, and any paper figures; never used for thresholds, early stopping, hyperparameter choice, or qualitative cherry-picking.

The split key must be scene/base-video level, not frame, mask, prompt, or generated-loser path. Exp25 already established VOR train/search/shadow split grouping with zero train-search, train-shadow, and search-shadow overlap, and VOR-Eval excluded. Exp27 should reuse that standard, including explicit overlap CSVs against final sets.

Required leakage checks:

- exact basename overlap across train/search/shadow/final;
- scene_group/base-video overlap across train/search/shadow/final;
- source archive member overlap for VOR-derived samples;
- caption/prompt ID overlap where generated captions are cached;
- loser-output path overlap, including resumed/partial generation directories;
- visual-case overlap against the final paper-case list.

Fail closed: if any overlap is found, the split is invalid and must be regenerated before training.

## BR/OR Protocol

Exp27 must not blur BR and OR semantics.

For OR/VOR:

- `condition = FG_BG / V_obj`;
- `winner = BG / V_bg`;
- `mask = foreground object mask`;
- `hard_comp = false`;
- loser = raw generator output unless the experiment explicitly names a different loser policy.

For BR:

- keep the existing BR evaluator path, mask convention, ProPainter prior path, raw6/no-PCM/no-blur settings, frame count, and metric stack fixed during comparison;
- do not use OR-derived gains to select BR checkpoints unless BR search-dev also passes.

The first Exp27 comparisons should report OR and BR separately. A combined score is acceptable only as a secondary table after per-protocol outcomes are shown.

## VOR-Eval Isolation

VOR-Eval is final-only. Exp25 extracted VOR-Eval as 43 aligned triplets and explicitly excluded it from train, selection, threshold, and checkpoint choice. Exp27 PRD repeats this guardrail. Therefore:

- no VOR-Eval loser generation for method selection;
- no VOR-Eval qualitative contact sheets before checkpoint lock;
- no VOR-Eval metric threshold tuning;
- no prompt/caption inspection from VOR-Eval for training design.

The only allowed pre-final VOR-Eval action is integrity verification that does not expose model outputs or guide choices.

## Variables One At A Time

Exp27 should use a factorial ladder with one changed variable per row. Minimum ladder:

1. `B0`: current LoVI/DiffuEraser baseline, unchanged data, unchanged objective, unchanged evaluator.
2. `B1`: LocalDPO-style data construction only, original DPO objective unchanged.
3. `B2`: region-aware/mask-local DPO loss only on the same `B1` pairs.
4. `B3`: add SFT/global DPO regularizer to `B2`, if `B2` passes.
5. `B4`: add SDPO winner-preserving loser-gradient scaling to `B2` or `B3`.
6. `B5`: replace sigmoid DPO weighting with Linear-DPO-style linear utility and EMA reference, with all data and masks fixed.

Do not combine LocalDPO data, mask-local loss, SDPO, Linear-DPO, data-scale increase, and optimizer changes in a single first run. That would be uninterpretable.

## Data Scale

The papers use much larger data than this project can safely assume: LocalDPO reports 63K high-quality real videos, Linear-DPO uses full Pick-a-Pic or a 210K HPDv3-sub for SD3, and SDPO uses large image preference corpora. Exp25 deliberately capped the first VOR source pool at 4096 train candidates, 256 search-dev, and 256 shadow-dev, with nested manifests at 512/1024/2048/3072 and 4096 only if 3072 clearly beats 2048.

Exp27 should follow the nested scale rule:

- first prove the data semantics on 128 examples;
- train/report nested 512, 1024, 2048, 3072;
- unlock 4096 only if 3072 beats 2048 on fixed search-dev and does not regress diagnostics;
- never change data scale and objective in the same promotion step.

Every scale row must reuse the same ordered nested manifest prefix so scale comparisons are paired rather than different random samples.

## Seeds And Determinism

Seeds must be explicit for:

- split generation;
- mask generation;
- loser generation;
- dataloader shuffling;
- diffusion timestep/noise sampling;
- inference/evaluation sampling.

LocalDPO official corruption generation uses random connected masks and a fixed pipeline generator seed; its dataset loader also samples frames/crops with NumPy randomness. Linear-DPO offsets seed by process index in training. These are reasonable implementation details, but Exp27 needs a run manifest recording all of them. For claims, use at least three seeds after a method passes the single-seed micro gate. Single-seed results can only promote to multi-seed confirmation, not to final claims.

## Fair Micro Gates

Borrow Exp26's ladder discipline:

- L0 official baseline strict-load and native inference reproduction.
- L1 native loss/noising/target parity.
- L2 policy=reference zero-gap test, DPO loss approximately `log(2)`.
- L3 one-step update, strict checkpoint save/reload, output changes.
- L4 10-step smoke on real preference data.
- L5 micro gates at 50/100/250/500 on fixed search-dev.
- L6 promote to 1000/1500/2000 only if the checkpoint curve improves.
- L7 final evaluation only after checkpoint lock.

Fairness constraints:

- same evaluator, seed, manifest, comp, mask, scheduler, output protocol;
- equal train steps for checkpoint comparisons;
- same inference wrapper and checkpoint-loading path for baseline and candidate;
- no VBench as an inpainting decision metric;
- report diagnostics: winner MSE, loser MSE, preference gap, implicit accuracy, loser-dominant rate, gradient norm, and mask-local versus global losses.

## LocalDPO-Style Data Baseline

Exp27 should include a LocalDPO-style data baseline before objective novelty. The adapted baseline should construct high-confidence OR pairs:

- winner: clean/background target from VOR/OR semantics;
- loser: same source after localized corruption or raw generator restoration in the masked object region;
- mask: stored and versioned;
- caption: fixed before split, not regenerated after seeing eval outputs;
- reference model: frozen baseline checkpoint used to generate losers.

This baseline is experimentally important because LocalDPO's central claim is data construction efficiency and high-confidence local preference, not merely the region-aware loss. A fair study must test whether simply replacing multi-sample/critic preferences with controlled local negatives improves the current pipeline.

Required ablations:

- current project loser data versus LocalDPO-style local-corruption losers;
- full-latent DPO versus mask-local DPO on identical pairs;
- with and without SFT anchor on winners;
- one corruption noise band at a time;
- one mask policy at a time.

## Objective Study Design

SDPO should be tested as a plug-in safeguard only after base data/loss parity is established. Its measurable claim is winner preservation, so the primary micro-gate should be whether winner loss avoids upward drift while search-dev metrics do not regress.

Linear-DPO should be tested as an objective weighting/reference update ablation only. Its measurable claim is sustained non-saturating updates, so the primary micro-gate should be reduced premature implicit-accuracy saturation without degrading search-dev image/video metrics. EMA reference introduces another variable and must be a separate ablation from the linear utility.

Do not claim Exp27 novelty for LocalDPO-style localized real-video preference construction, SDPO winner preservation, or Linear-DPO linear utility/EMA reference. Novelty, if any, must be framed as task adaptation to video inpainting/OR under strict split isolation.

## Required Report Tables

Exp27 should produce these before any final evaluation:

- split-overlap audit table;
- data-scale nested table;
- objective ablation table with one changed variable per row;
- seed table for promoted methods;
- micro-gate pass/fail table;
- BR and OR metric tables separated;
- diagnostics table showing winner/loser/mask-local dynamics;
- final-only audit confirming VOR-Eval/DAVIS50/YouTubeVOS100 were untouched before checkpoint lock.

## Go / No-Go

Go for Exp27 design and micro gates if the split manifest and L0-L3 parity tests are complete.

No-go for long training until:

- VOR-Eval isolation is mechanically enforced;
- train/search/shadow/final overlap audits pass;
- BR/OR manifests are semantically validated;
- LocalDPO-style data baseline is implemented as its own row;
- each objective change has a single-variable ablation;
- search-dev gates pass before shadow-dev and final evaluation.

