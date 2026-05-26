# Experiment Matrix

## Core Ablation Directions This Week

The full paper-story plan has 9 experiment directions: 3 completed and 6 new.
The current省时版 generation source is `diffueraser_only`; four generator
models are not expanded into four training experiments in the first-round main
line. Do not describe the active D1/D2 plan as all-models source.

Completed:

1. `diffueraser_reproduction_sft`
2. `official_videodpo_vc2`
3. `official_videodpo_diffueraser`

New:

4. `official_videodpo_diffueraser_data_fullmask_loser`
5. `official_videodpo_diffueraser_data_partialmask_loser_comp_k4`
6. `official_videodpo_diffueraser_data_partialmask_loser_nocomp_k4`
7. `official_videodpo_diffueraser_task_partialmask_samemask_fullloss`
8. `official_videodpo_diffueraser_task_partialmask_samemask_regionloss`
9. `official_videodpo_diffueraser_youtubevos_partialmask_data`

The important boundary:

- Experiments 4, 5, and 6 are **data-only ablations**. Masks are used only to generate offline losers. Training still uses the completed `official_videodpo_diffueraser` full-mask bridge, so the training model does not receive partial masks.
- Experiments 7 and 8 are the first **task ablations**. The partial mask becomes a training-time model input, so DiffuEraser actually performs partial video inpainting.
- Experiment 9 is a **data-source ablation** on top of the better experiment 7/8 partial-mask task setting, moving from VideoDPO data to YouTube-VOS-derived data.

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
| `official_videodpo_diffueraser_data_fullmask_loser` | diagnostic/failure case; no full generation | DiffuEraser bridge | VideoDPO | VideoDPO winner | full-mask inpainting generated | full | full | false | offline | data diagnostic | PSNR, SSIM, VBench, SBS, DPO diagnostics | Old H20-2 OR root is invalid and retained only for failure audit: `/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_fullmask_loser`. BR/no-prior root technically passed 100-row manifest/decode checks but failed quality gate: 95/100 `too_bad`, median q=`0.1947`, max q=`0.3315`. Do not train experiment 4 from this D1 unless it is explicitly a failure ablation. |
| `official_videodpo_diffueraser_data_partialmask_loser_comp_k4` | D2 ready; training entrypoint ready | DiffuEraser bridge | VideoDPO | VideoDPO winner | partial-mask inpainting + composite | partial K=4 | full | true | offline | data | PSNR, SSIM, VBench, SBS, DPO diagnostics | PAI output root: `/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_data_partialmask_loser_k4`; final readiness passed with `sampled=100`, `sample_issues=0`, `ready=True`; use `selected_primary_comp.repaired.jsonl`; `generation_source=diffueraser_only`; `diffueraser_inference_stack=or`; `diffueraser_prior_mode=propainter`. |
| `official_videodpo_diffueraser_data_partialmask_loser_nocomp_k4` | D2 ready; training entrypoint ready | DiffuEraser bridge | VideoDPO | VideoDPO winner | partial-mask raw output | partial K=4 | full | false | offline | data diagnostic | PSNR, SSIM, VBench, SBS, DPO diagnostics | Reuses D2 raw loser manifest `selected_primary_nocomp.repaired.jsonl`; no new model inference. |
| `official_videodpo_diffueraser_task_partialmask_samemask_fullloss` | dataset entrypoint ready; 5-step smoke pending on PAI | DiffuEraser | generated partial-mask data | VideoDPO winner | partialmask comp loser | partial | same partial mask | true | offline data | task | metric.py, SBS, DPO diagnostics | Experiment 7. Uses `selected_primary_comp.repaired.jsonl` and `mask_path`; first mask policy: `M_train = M_gen`; loss remains full-video DPO loss. |
| `official_videodpo_diffueraser_task_partialmask_samemask_regionloss` | dataset entrypoint ready; region loss not implemented | DiffuEraser | generated partial-mask data | VideoDPO winner | partialmask comp loser | partial | same partial mask | true | offline data | task + loss | metric.py, SBS, DPO diagnostics | Experiment 8. Reuses D2 comp data and masks; dataset/CLI is ready, but region weighting must be implemented as a wrapper around the existing DPO loss before training. |
| `official_videodpo_diffueraser_youtubevos_partialmask_data` | readiness only; no full generation | DiffuEraser | YouTube-VOS | YouTube-VOS clean/target clip | partial-mask generated loser | partial | partial | true first | offline | data source | PSNR, SSIM, VBench, SBS, DPO diagnostics | H20 train split confirmed under `ytbv_2019_full_resolution/train`; current prompt policy is `PROMPT_MODE=none`, `prompt_source=no_prompt`; final D3 waits until experiments 7/8 choose the task/loss setting. |
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

## Generation Source Semantics

The active D2 production run and D1 diagnostic validation use
`generation_source=diffueraser_only`. This label is about the manifest-level
source model: only DiffuEraser output is used as a candidate and selected
loser. It does not mean every DiffuEraser code path disables ProPainter
internally.

Current DiffuEraser OR inference runs:

1. ProPainter produces a prior video, saved in the work directory as
   `propainter.mp4`.
2. DiffuEraser consumes the original input video, mask, prompt, and the
   ProPainter prior.
3. The final generated loser is the DiffuEraser result, saved as
   `diffueraser.mp4` and recorded with `generation_model=diffueraser`.

Therefore `diffueraser_only` is not a propainter-only or all-model source. For
D2 it is also not a no-prior DiffuEraser ablation, because partialmask currently
keeps OR + ProPainter prior. D1 BR/no-prior is a separate fullmask diagnostic
path with `diffueraser_prior_mode=noise`.

For D1 specifically, OR is no longer approved: a full-frame mask removes all
visible context, which can make OR/ProPainter prior and DiffuEraser refinement
degenerate into blurry or abstract output. BR/no-prior was tested at limit=100
and still failed quality, so D1 is not a first-round training dataset. D2
partialmask is not the same risk profile because it uses local masks and can
composite generated regions back into the original winner.
