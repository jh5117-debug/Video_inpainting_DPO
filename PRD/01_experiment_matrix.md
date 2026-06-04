# Experiment Matrix

## 2026-06-04 Naming And Artifact Registry Repair

This matrix must be read together with:

- `PRD/12_experiment_artifact_registry.md`
- `PRD/13_dpo_diag_audit.md`
- `PRD/14_pai_manual_artifact_search_commands.md`

Naming is now fixed:

| Presentation name | Correct experiment identity | Notes |
| --- | --- | --- |
| Exp4 | `official_videodpo_diffueraser_data_fullmask_loser` / full-mask generated-loser data-only smoke | Data changed only: VideoDPO winner stays winner; loser becomes DiffuEraser full-mask generated video. Task remains full-mask/video-generation bridge, not partial-mask reconstruction. |
| Old Exp5 | `exp5_d2_comp_k4_stage1/stage2_full` and `exp5_d2_comp_k4_beta10_s1s2_4000` | Collapsed diagnostic runs. Ranking can look correct while visual quality is broken. |
| New Exp5 | `exp5_d2_comp_k4_wingap_lose025_beta10_s1s2_4000` | Winner-anchored comp data-only rerun. Separate from Old Exp5. |
| New Exp6 | `exp6_d2_nocomp_k4_wingap_lose025_beta10_s1s2_4000` | No-comp + changed loss diagnostic. This is **not** an ordinary standalone Exp6. |
| Exp7 | `exp7_d2_comp_k4_partial_wingap_lose025_beta10_s1s2_gate1500` | First partial-mask task-alignment gate. |
| Exp8 | `exp8_youtubevos_d3_comp_regionloss_wingap_lose025_stage1_gate1500` | Region-loss target-domain diagnostic; do not claim completed without PAI artifacts. |
| Exp9 | D3 target-domain partial-mask gates | H20 nocomp and no-lose artifacts found; PAI clean comp still needs PAI artifact scan. |

Artifact requirement for every row:

- independent folder;
- dpo-diag CSV for DPO runs;
- qualitative videos or screenshots;
- eval report and metrics;
- checkpoint folder if training completed.

Rows that lack any of those are `artifact gap` / `diag gap`, not complete.

## 2026-05-31 Exp5 Collapse And Winner-Anchored Rerun

Old `exp5_d2_comp_k4_stage1/stage2_full` with `beta_dpo=500` and 10000-step
Stage1/Stage2 is now marked **failed / collapsed / diagnostic only**.

This failure is not a task failure: Exp3 showed the VideoDPO-to-DiffuEraser DPO
bridge can work. The old Exp5 failure is an optimization/preference-data
failure caused by D2 generated losers plus full-mask training, full-video loss,
`beta_dpo=500`, no SFT regularization, and long training. Early `acc=1`,
`dpo=0`, and `loss=0` are treated as DPO saturation, not as visual success.

The replacement reruns are:

| Experiment | Status | Manifest | beta_dpo | Stage1 steps | Stage2 steps | Validation during training | Post Stage2 eval |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `exp5_d2_comp_k4_wingap_lose025_beta10_s1s2_4000` | planned/running | `selected_primary_comp.repaired.jsonl` | 10 | 4000 | 4000 | disabled | qual30 + full VBench |
| `exp6_d2_nocomp_k4_wingap_lose025_beta10_s1s2_4000` | planned/running | `selected_primary_nocomp.repaired.jsonl` | 10 | 4000 | 4000 | disabled | qual30 + full VBench |

The intermediate `exp5_d2_comp_k4_beta10_s1s2_4000` rerun is also marked
**failed / collapsed / diagnostic only**. Stage2 loaded Stage1 correctly, but
diagnostics showed `mse_w >> ref_mse_w`, `mse_l >> ref_mse_l`, near-saturated
`implicit_acc`, and low DPO loss while qualitative outputs became universal
stripe/high-frequency textures. This confirms the problem is the unanchored DPO
objective, not a launcher or evaluation bug.

Old H20 Exp6 unanchored training is superseded by the winner-anchored rerun and
should be stopped if still running.

Winner-anchored rerun parameters:

```text
beta_dpo = 10
winner_abs_reg_weight = 0.05
winner_gap_reg_weight = 1.0
winner_gap_reg_margin = 0.0
lose_gap_weight = 0.25
sft_reg_weight = 0.0
stage1_steps = 4000
stage2_steps = 4000
```

## 2026-05-31 Exp7 Partial-Mask Task Gate

Exp5 winner-anchored improved the failure mode but is not final: the winner
anchor held `win_gap` down, yet qualitative outputs still show texture/color
attractors under the data-only full-mask/full-video bridge. Exp6 no-comp is
running on H20 and must continue for the comp-vs-no-comp comparison.

Exp7 is the next gate and is a **task ablation**, not a data-only run. It keeps
the D2 comp manifest but changes the training task so DiffuEraser sees the same
partial mask used during loser generation.

| Experiment | Status | Manifest | Train mask | Mask source | beta_dpo | lose_gap_weight | winner_abs_reg | winner_gap_reg | Stage1 | Stage2 | Gate validation |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `exp7_d2_comp_k4_partial_wingap_lose025_beta10_s1s2_gate1500` | planned / launching on PAI | `selected_primary_comp.repaired.jsonl` | partial | manifest `mask_path` | 10 | 0.25 | 0.05 | 1.0 | 1500 | 1500 | qual30 SBS + DPO diag summary |

Exp7 interpretation rule:

- If Exp7 gate is more stable than Exp5, D2 generated losers are not the sole
  problem; the data-only full-mask objective was mismatched with the partial
  mask generation process.
- If Exp7 gate still collapses, the next change should be data/prompt quality
  or a stronger winner-preservation strategy, not a direct full 4000+4000 run.

Exp7 gate1500 result update:

| Experiment | Current status | Observed eval | Interpretation | Next action |
| --- | --- | --- | --- | --- |
| `exp7_d2_comp_k4_partial_wingap_lose025_beta10_s1s2_gate1500` | inconclusive / risky | full-mask qual30 failed; stripe-heavy; some samples worse than new Exp5 | task-mismatched eval because training is partial-mask inpainting | run true partial-mask manifest eval before deciding failure |

Observed diagnostics:

- `winner_gap_reg_weight=1.0` keeps `win_gap` relatively bounded.
- `loser_dominant_ratio` can reach 1.0 for late steps.
- `mse_l_over_ref_mse_l` can become very high, so loser degradation remains a
  strong shortcut even with winner anchoring.

Prepared but not launched:

| Experiment | Status | Purpose |
| --- | --- | --- |
| `exp7_d2_comp_k4_partial_wingap_nolose_beta10_s1s2_gate1000` | script prepared only | cut `lose_gap_weight` to 0.0 if partial-mask eval confirms loser-degradation shortcut |

## Core Ablation Directions This Week

The core plan has four directions, with Direction 2 split into 2A/2B:

1. `official_videodpo_diffueraser_data_fullmask_loser`
2. `official_videodpo_diffueraser_data_partialmask_loser_comp_k4`
3. `official_videodpo_diffueraser_data_partialmask_loser_nocomp_k4`
4. `official_videodpo_diffueraser_task_partialmask`
5. `official_videodpo_diffueraser_youtubevos_partialmask_data`

The important boundary:

- Experiments 1, 2A, and 2B are **data-only ablations**. Masks are used only to generate offline losers. Training still uses the completed `official_videodpo_diffueraser` full-mask bridge, so the training model does not receive partial masks.
- Experiment 3 is the first **task ablation**. The partial mask becomes a training-time model input, so DiffuEraser actually performs partial video inpainting.
- Experiment 4 is a **data-source ablation** on top of the Experiment 3 partial-mask task setting, moving from VideoDPO data to YouTube-VOS-derived data.

## Priorities

| Priority | Work | Notes |
| --- | --- | --- |
| 0 | Protect completed experiments | Do not break `official_videodpo_vc2`, `official_videodpo_diffueraser`, or DiffuEraser reproduction/SFT scripts. |
| 1 | PAI audit | Confirm model envs/weights, data paths, and completed output artifacts. |
| 2 | Full-mask loser data ablation | Data-only; win is VideoDPO winner, loser is full-mask inpainting output. |
| 3 | Partial-mask offline + comp | First priority for partial masks; cleanest data-only ablation. |
| 4 | Partial-mask offline + no-comp | Second priority; diagnostic comp-vs-no-comp ablation. |
| 5 | Partial-mask training task | Task ablation; partial mask becomes training condition. |
| 6 | YouTube-VOS partial-mask data | Data source ablation built on partial-mask task setting. |
| Future | Online loser generation | Not first version; document only until offline generation is stable. |

## Matrix

| Experiment | Status | Model | Data source | Win source | Loser source | Mask for loser generation | Mask for training | Comp | Offline/Online | Changed variable | Metrics | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `diffueraser_reproduction_sft` | completed | DiffuEraser | DAVIS / YouTube-VOS-derived | source video | reconstruction/inpainting output | task-specific | task-specific | setting-dependent | offline | reproduction/SFT/metric setting | PSNR, SSIM, VBench | Best eval: 6 steps, no PCM, no Gaussian blur, frame-wise metric transfer. |
| `official_videodpo_vc2` | completed | VC2 | VideoDPO | VideoDPO winner | VideoDPO rejected | none | none | false | existing pairs | official baseline | VBench, SBS | Completed full VBench. |
| `official_videodpo_diffueraser` | completed | DiffuEraser | VideoDPO | VideoDPO winner | VideoDPO rejected | none | full | false | existing pairs | model adapter | VBench, SBS, DPO diagnostics | Official VideoDPO skeleton + DiffuEraser full-mask bridge. |
| `official_videodpo_diffueraser_data_fullmask_loser` | asset-ready, smoke pending | DiffuEraser bridge | VideoDPO | VideoDPO winner | full-mask inpainting generated | full | full | false | offline | data | PSNR, SSIM, VBench, SBS, DPO diagnostics | Data/weight paths found; run one-sample generation smoke before full generation. |
| `official_videodpo_diffueraser_data_partialmask_loser_comp_k4` | asset-ready, smoke pending | DiffuEraser bridge | VideoDPO | VideoDPO winner | partial-mask inpainting + composite | partial K=4 | full | true | offline | data | PSNR, SSIM, VBench, SBS, DPO diagnostics | Cleanest partial-mask data-only ablation; wait for generator smoke. |
| `official_videodpo_diffueraser_data_partialmask_loser_nocomp_k4` | asset-ready, smoke pending | DiffuEraser bridge | VideoDPO | VideoDPO winner | partial-mask raw output | partial K=4 | full | false | offline | data diagnostic | PSNR, SSIM, VBench, SBS, DPO diagnostics | Reuses the same raw generation as comp; wait for generator smoke. |
| `official_videodpo_diffueraser_task_partialmask` | scaffold | DiffuEraser | generated partial-mask data | VideoDPO winner | partialmask comp loser | partial | partial | true | offline data | task | PSNR, SSIM, VBench, SBS, DPO diagnostics | First mask policy: same-mask. |
| `official_videodpo_diffueraser_youtubevos_partialmask_data` | path-confirmed scaffold | DiffuEraser / generator models | YouTube-VOS | YouTube-VOS clean/target clip | partial-mask generated loser | partial | partial | true first | offline | data source | PSNR, SSIM, VBench, SBS, DPO diagnostics | PAI train split confirmed under `ytbv_2019_full_resolution/train`; prompt policy still needs definition. |
| `official_videodpo_diffueraser_online_loser_generation` | future | TBD | TBD | TBD | generated during training | TBD | TBD | TBD | online | generation timing | TBD | Not first priority. |

## Offline / Online And Comp / No-Comp

First priority: `offline + comp`.

- Reproducible.
- Training speed is stable.
- Generation cost does not leak into DPO training time.
- Win and loser are identical outside the mask, giving the cleanest control variable.

Second priority: `offline + no-comp diagnostic`.

- Useful to determine whether compositing is necessary.
- Can introduce mask-outside color, texture, brightness, temporal, or background drift.
- Should not replace the comp main experiment.

Not first version: `online loser generation`.

- More diverse negatives, but costly and stochastic.
- Couples generation with training.
- Harder to debug than offline manifest-driven data.
