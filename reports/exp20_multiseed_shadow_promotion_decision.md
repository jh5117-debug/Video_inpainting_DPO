# Exp20 Multi-Seed Shadow Promotion Decision

- pre-visual decision: `NO_CANDIDATE_PROMOTED_PRE_VISUAL`
- final decision after Codex visual review: `NO_CANDIDATE_PROMOTED`
- final status: `COMPLETED_NEGATIVE_AFTER_MULTISEED_SHADOW`
- enter 500-step gate: `False`
- stopped actions: no 500-step, no 1000/2000-step, no Stage2, no DAVIS50, no YouTubeVOS100.

| candidate | pre-visual gate | search delta vs Exp11-S1 | shadow delta vs Exp11-S1 | loser delta vs P0 |
|---|---:|---:|---:|---:|
| P4 | False | 0.044370 | -0.084620 | -0.033333 |
| BF07 | False | 0.046550 | -0.128949 | -0.166667 |

## BF07 Replaces P4 Checks

- search_mean_psnr_ge_p4_plus_0p01: `False`
- shadow_mean_psnr_ge_p4_plus_0p01: `False`
- seed_wins_ge_2: `False`
- shadow_boot_prob_ge_0p95: `False`
- shadow_per_video_win_rate_ge_0p55: `False`
- lpips_degrade_le_0p0002: `False`
- vfid_degrade_le_0p003: `False`
- tc_drop_le_0p00015: `False`
- ewarp_not_worse: `True`
- loser_degrade_not_worse_than_p4_plus_0p05: `True`

## Visual Review Result

The visual review covered all 20 shadow-dev anonymous contact sheets.

| Visual judgement | Count |
|---|---:|
| BF07 visibly better than P4 | 0 |
| P4 visibly better than BF07 | 4 |
| Tie / no reliable difference | 16 |
| BF07 new obvious artifact | 0 |
| P4 new obvious artifact | 0 |

BF07's small search-dev PSNR/Ewarp signal does not translate into visible improvement on shadow-dev. In several thin-structure or high-motion cases BF07 is slightly softer than P4. P4 also does not show a stable visual advantage over Exp11-S1 on shadow-dev.

## AD04 Conditional Reference

EQ_AD04 was evaluated once on shadow-dev as the preregistered adaptive reference. It was close to P4 in PSNR and slightly better in LPIPS/VFID, but TC was lower than P4 and it was not the main P4/BF07 decision target. It was therefore kept as a single-seed adaptive ablation and not expanded to seeds 20260620/20260621.

## Final Decision

Neither P4 nor BF07 passes the Stage1 promotion gate:

- P4 search-dev mean exceeds Exp11-S1, but shadow-dev mean is `-0.084620 dB` below shadow Exp11-S1.
- BF07 search-dev mean is similar to P4, but shadow-dev mean is `-0.128949 dB` below shadow Exp11-S1 and `-0.044330 dB` below P4.
- BF07 fails the replacement gate for P4 on search mean, shadow mean, seed wins, bootstrap probability, per-video win rate, LPIPS, VFID, and TC.
- Visual review does not provide a counter-signal.

Final decision:

`NO_CANDIDATE_PROMOTED`

Exp20 stopped after multiseed shadow confirmation. No more boundary sweep is recommended.
