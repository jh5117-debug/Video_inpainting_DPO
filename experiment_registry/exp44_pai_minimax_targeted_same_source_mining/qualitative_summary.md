# Exp44 Qualitative Summary

Readback imports Exp42 visual finding: real MiniMax successful-removal signal exists, but row-level success/failure labels are source-clustered and failure-label noisy. Exp44 must use targeted same-source mining plus strict visual relabeling before any Stage2 dataset handoff.

## 2026-06-29 Targeted Source Manifest

The locked mining manifest keeps only A/B/C groups from the preregistered plan:
existing-overlap groups, success-only groups needing failures, and failure-only
groups needing successes. Fallback groups are still held back. This prepares
official MiniMax raw inference, but no new video evidence has been generated yet.

## 2026-06-29 Targeted Second-Pass Mining

Official MiniMax raw inference completed on the locked A/B/C source manifest.
This produced a much denser automatic candidate pool than Exp42: automatic
same-source pair capacity rose from Exp42's `7` overlap groups to an auto-level
capacity of `26` pairs across `13` groups. However, no qualitative promotion is
made yet. The automatic success/failure labels are known to be noisy from Exp42,
and Exp44's next required milestone is strict visual relabeling of the selected
candidates before any pair, bad-noise, or Stage2 handoff dataset is trusted.

No training, optimizer step, VOR-Eval use, hard comp, H20 modification, or
MiniMax third-backbone-positive claim occurred.

## 2026-06-29 Strict Visual Relabeling

Codex opened and inspected all `47` selected candidate review pages before
relabeling. The visual pass confirmed that automatic metrics over-count clean
success: many rows are useful only as bounded failures or pseudo-success targets
because of residual objects, water/reflection artifacts, slight geometry
breaks, or local smearing.

Conservative qualitative outcome:

- clean success exists, especially in simple indoor, snow, hallway/stair, and
  stable-road cases;
- usable pseudo-success exists but should remain separate from GT targets;
- medium-hard failures are plentiful and mostly local, with outside preservation
  sufficient for DPO loser construction;
- severe fogging, over-erasure, boundary destruction, outside damage, and
  too-close rows were rejected.

The result is a cleaner source for same-source pair construction, not evidence
that MiniMax DPO/SFT has improved model quality. No training or optimizer step
occurred.

## 2026-06-29 Same-Source Pair Construction

The constructed pairs preserve the intended MiniMax-native signal:

- winner for DPO is GT background, not a generated pseudo target;
- pseudo-success is carried separately for possible Stage2 distillation;
- loser is a visually relabeled medium-hard MiniMax raw output from the same
  source group;
- no cross-source matching is used;
- train/search/shadow scene groups are disjoint.

This is the first Exp44 point where the same-source minimum pair gate passes
after human visual relabeling. It is still a data milestone, not a model-quality
milestone.

## 2026-06-29 Bad-Noise v4 State Construction

The bad-noise v4 records preserve the same-source MiniMax-native signal:

- condition and loser come from the same source group;
- winner remains GT background for preference fields;
- pseudo-success is retained separately for Stage2 distillation metadata;
- H-state tags identify local failure hard cases and winner-safe hard cases;
- outside-risk and winner-risk bounds are recorded before any future runner can
  consume the data.

This makes Exp44 ready to assemble a Stage2-style handoff package. It does not
claim that MiniMax training has improved: no training, optimizer step,
VOR-Eval use, hard comp, or H20 modification occurred.

## 2026-06-29 Stage2-Style Dataset Handoff

The Stage2 handoff separates three signals instead of mixing them:

- GT distillation keeps `V_bg` as the target;
- pseudo-success distillation uses only visually approved MiniMax
  `SUCCESS_CLEAN` / `SUCCESS_USABLE` outputs and is the recommended first H20
  30-step debug run;
- same-source preference keeps GT as the default winner and same-source
  `FAILURE_MEDIUM_HARD` output as loser, with bad-noise v4 metadata attached.

This is partial because the split is only `24/8/8`, below the `32/16/16`
minimum. H20 must verify absolute NAS paths before running because this current
Codex session did not have `/mnt/nas` mounted. No quality-positive claim is
made.
