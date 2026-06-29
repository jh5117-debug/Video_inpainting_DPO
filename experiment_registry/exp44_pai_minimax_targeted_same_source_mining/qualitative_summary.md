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
