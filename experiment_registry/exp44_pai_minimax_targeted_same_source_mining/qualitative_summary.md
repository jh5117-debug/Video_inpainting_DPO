# Exp44 Qualitative Summary

Readback imports Exp42 visual finding: real MiniMax successful-removal signal exists, but row-level success/failure labels are source-clustered and failure-label noisy. Exp44 must use targeted same-source mining plus strict visual relabeling before any Stage2 dataset handoff.

## 2026-06-29 Targeted Source Manifest

The locked mining manifest keeps only A/B/C groups from the preregistered plan:
existing-overlap groups, success-only groups needing failures, and failure-only
groups needing successes. Fallback groups are still held back. This prepares
official MiniMax raw inference, but no new video evidence has been generated yet.
