# New Exp6 Prompt-Length And Exp9 Nocomp Plan

Updated: 2026-06-04 with naming and artifact repair.

## Naming Boundary

This document uses **New Exp6** only.

New Exp6 means:

```text
exp6_d2_nocomp_k4_wingap_lose025_beta10_s1s2_4000
```

It is not a generic Exp6. It is the no-comp diagnostic counterpart to New Exp5:

- data: reuses D2 partial-mask raw loser outputs;
- manifest: nocomp, `final_loser = raw_loser`;
- task: still data-only full-mask bridge, not partial-mask reconstruction;
- loss: winner-anchored DPO with `beta=10`, `winner_abs=0.05`,
  `winner_gap=1.0`, `lose_gap=0.25`.

## Current Artifact Status

H20 audit found:

- New Exp6 training dpo-diag:
  `experiments/dpo/stage1/20260601_004753_exp6_d2_nocomp_k4_wingap_lose025_beta10_s1s2_4000_stage1/dpo_diagnostics.csv`
- New Exp6 stage2 dpo-diag:
  `experiments/dpo/stage2/20260601_004753_exp6_d2_nocomp_k4_wingap_lose025_beta10_s1s2_4000_stage2/dpo_diagnostics.csv`
- New Exp6 prompt-length analysis folder:
  `logs/analysis/new_exp6_prompt_length`
- Local qualitative videos:
  `/home/hj/dpo-2-1-exp/new-exp6`

Artifact gaps:

- Any PAI-side eval reports and complete qual/VBench summaries still need PAI
  manual search if they were produced there.

## Interpretation

New Exp6 is useful because it separates comp from no-comp:

- It does not generate new videos.
- It reuses Exp5 raw losers and changes the manifest.
- It tests whether the comp operation itself introduced harmful artifacts.

Observed qualitative note:

- Some long-prompt samples looked better than base or New Exp5.
- This is not enough to call it final success without dpo-diag + metric summary.
- It remains a data-only full-mask bridge, so it does not solve the final
  partial-mask inpainting task mismatch.

## Exp9 Nocomp Link

Exp9 nocomp moves the no-comp idea to D3/YouTube-VOS target-domain data and
partial-mask inpainting. H20 audit found:

```text
experiments/dpo/stage1/20260603_131758_exp9_youtubevos_d3_nocomp_partialmask_wingap_lose025_stage1_gate1500_h20_stage1
logs/target_eval/exp9_d3_nocomp_gate_h20_20260604_023243
```

Exp9 nocomp should be compared against PAI clean Exp9 comp only after PAI
returns complete artifact search outputs.

