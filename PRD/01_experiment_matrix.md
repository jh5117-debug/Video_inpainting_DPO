# Experiment Matrix

## Core Ablation Directions This Week

The full paper-story plan has 9 experiment directions: 3 completed and 6 new.
The current省时版 generation source is `diffueraser_only`; four generator
models are not expanded into four training experiments in the first-round main
line.

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
| `official_videodpo_diffueraser_data_fullmask_loser` | D1 H20 audit/smoke pending | DiffuEraser bridge | VideoDPO | VideoDPO winner | full-mask inpainting generated | full | full | false | offline | data | PSNR, SSIM, VBench, SBS, DPO diagnostics | `generation_source=diffueraser_only`; prepare H20 small sample before full generation. |
| `official_videodpo_diffueraser_data_partialmask_loser_comp_k4` | D2 PAI generation running | DiffuEraser bridge | VideoDPO | VideoDPO winner | partial-mask inpainting + composite | partial K=4 | full | true | offline | data | PSNR, SSIM, VBench, SBS, DPO diagnostics | `generation_source=diffueraser_only`; cleanest partial-mask data-only ablation. |
| `official_videodpo_diffueraser_data_partialmask_loser_nocomp_k4` | reuses D2 data | DiffuEraser bridge | VideoDPO | VideoDPO winner | partial-mask raw output | partial K=4 | full | false | offline | data diagnostic | PSNR, SSIM, VBench, SBS, DPO diagnostics | Reuses D2 raw loser manifest; no new model inference. |
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
