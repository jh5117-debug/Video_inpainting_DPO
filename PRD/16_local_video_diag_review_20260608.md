# Local Video And DPO-Diag Review

Updated: 2026-06-08

Scope:

- Local result root: `/home/hj/dpo-2-1-exp`
- Videos reviewed: 137
- Static images reviewed: 4
- DPO diagnostic CSVs reviewed: 9 experiment CSVs plus `index.csv`
- Review artifacts generated locally: `/home/hj/dpo-2-1-exp/_review`

This note records qualitative and diagnostic conclusions only. It intentionally
does not propose fixes or new methods.

## Inventory

Video groups reviewed:

| Group | Videos | Meaning in this review |
| --- | ---: | --- |
| `exp5` | 7 | old video-generation style baseline/comparison |
| `new-exp5` | 6 | New Exp5 D2 comp regularized DPO |
| `new-exp6` | 9 | New Exp6 D2 no-comp regularized DPO |
| `exp7a-1` | 20 | Exp7a Stage1-DPO + Stage2-SFT small-D2/VideoDPO eval |
| `exp7a-2` | 23 | Exp7a Stage1-DPO + Stage2-DPO small-D2/VideoDPO eval |
| `exp8a-1` | 18 | Exp8a Stage1-DPO + Stage2-SFT DAVIS eval |
| `exp8a-2` | 16 | Exp8a Stage1-DPO + Stage2-DPO DAVIS eval |
| `exp8c-1` | 17 | Exp8c GT-win Stage1-DPO + Stage2-SFT DAVIS eval |
| `exp8c-2` | 21 | Exp8c GT-win Stage1-DPO + Stage2-DPO DAVIS eval |

DPO diagnostic files reviewed:

- `/home/hj/dpo-2-1-exp/dpo-diag/exp5_new_stage1_dpo_diagnostics.csv`
- `/home/hj/dpo-2-1-exp/dpo-diag/exp5_new_stage2_dpo_diagnostics.csv`
- `/home/hj/dpo-2-1-exp/dpo-diag/exp6_stage1_dpo_diagnostics.csv`
- `/home/hj/dpo-2-1-exp/dpo-diag/exp7a-1_stage1_dpo_diagnostics.csv`
- `/home/hj/dpo-2-1-exp/dpo-diag/exp7a-2_stage2_dpo_diagnostics.csv`
- `/home/hj/dpo-2-1-exp/dpo-diag/exp8a_1_stage1_dpo_diagnostics.csv`
- `/home/hj/dpo-2-1-exp/dpo-diag/exp8a_2_stage2_dpo_diagnostics.csv`
- `/home/hj/dpo-2-1-exp/dpo-diag/exp8c_1_stage1_dpo_diagnostics.csv`
- `/home/hj/dpo-2-1-exp/dpo-diag/exp8c_2_stage2_dpo_diagnostics.csv`

## Review Method

All local videos were opened and sampled across time. For each video, review
contact sheets were generated at 8%, 28%, 50%, 72%, and 92% of the timeline.
For side-by-side videos, an additional mid-frame column split was generated to
separate `winner/GT`, `mask overlay`, `DiffuEraser-base`, and `current` columns.

Generated local review files:

- `/home/hj/dpo-2-1-exp/_review/contact_<group>.jpg`
- `/home/hj/dpo-2-1-exp/_review/cols_<group>.jpg`
- `/home/hj/dpo-2-1-exp/_review/video_metrics.csv`

The color statistics in `video_metrics.csv` were used only as an auxiliary
signal for purple/magenta artifacts. Final conclusions are based on visual
inspection plus DPO diagnostics.

## Overall Conclusion

The video-generation-style experiments and the mask-inpainting experiments are
showing two different behaviors.

For the video-generation-style setting, DPO can improve visual clarity and
semantic quality. This is most visible in `new-exp5` and partly in `new-exp6`:
the DPO output is often clearer, more object-like, and more scene-consistent
than the base/comparison side.

For the partial-mask inpainting setting, DPO is often not improving natural
completion. Instead, the model frequently places artificial white, pink, purple,
black-purple, grid-like, or patch-like structures inside the mask region. This
artifact is mask-region-bound and visually different from ordinary low-quality
generation.

The strongest diagnostic mismatch is that failing inpainting samples often have
strong DPO diagnostic values. High `implicit_acc`, low `dpo_loss`, large
`lose_gap`, and large `mse_l_over_ref_mse_l` indicate that the objective learned
to separate winner from loser, but that separation does not reliably correspond
to realistic inpainting.

## Per-Experiment Conclusions

### Exp5

`exp5` is a failed old setting. The current side is dominated by black-white
high-frequency stripe artifacts. Semantic video content is mostly lost. This is
not evidence that DPO improved video generation; it is a collapse example.

### New Exp5

`new-exp5` is the clearest positive video-generation result.

The base side often contains blocky texture, pseudo-text, or dense high-frequency
structure. The current side recovers recognizable scenes and subjects: beach,
Seine/Eiffel Tower, panda, boat, and dog examples are visibly clearer and more
semantically coherent. This matches the earlier observation that, when the task
is treated as video generation, DPO after the regularized loss can make outputs
clearer and more aesthetically coherent.

DPO diagnostics:

- Stage1 and Stage2 both run to 4000 steps.
- Final rows can return near neutral around `dpo_loss ~= 0.693`, but the last
  windows still show substantial intermittent separation.
- This diagnostic pattern is consistent with visible improvement but not a
  perfectly stable monotonic objective trace.

Conclusion: `new-exp5` is a useful positive qualitative example for
video-generation-style DPO, not for mask inpainting.

### New Exp6

`new-exp6` is also mostly positive but less clean than `new-exp5`.

Many current outputs are more recognizable than the base side: the boat,
panda, Iron Man, and pixel-art panda examples show clearer subject/scene
structure. However, some samples still retain strong high-frequency texture
noise, especially beach and boat/raccoon-like cases.

DPO diagnostics:

- Only a short Stage1 diagnostic CSV is available locally, so the diag evidence
  is not comparable to the full 2000/4000-step runs.

Conclusion: `new-exp6` supports the same broad direction as `new-exp5`, but with
more residual texture artifacts and weaker diagnostic completeness.

### Exp7a-1: Stage1-DPO + Stage2-SFT

`exp7a-1` is the first clear partial-mask failure family in the reviewed local
folder.

The videos are small-D2/VideoDPO side-by-side comparisons. The current column
already shows white/pink grid patches, bright mask-shaped regions, and
unnatural local objects. Some examples remain usable, but the failure mode is
obvious in multiple samples.

Auxiliary purple/magenta statistics:

- 20 videos reviewed.
- 9 videos have center purple max above 0.05.
- 6 videos have center purple max above 0.10.
- Severe examples include `000085`, `000523`, `000365`, `000615`, `000467`, and
  `000450`.

DPO diagnostics:

- Final `implicit_acc = 1.0`.
- Final `lose_gap = 0.4505`.
- Final `mse_l_over_ref_mse_l = 52.26`.
- Last-10 mean `mse_l_over_ref_mse_l = 93.32`.
- Max `mse_l_over_ref_mse_l = 301.6`.

Conclusion: Exp7a Stage1 learned strong loser separation, but the visual result
already contains mask-bound artifacts. The DPO diagnostics do not guarantee
inpainting quality.

### Exp7a-2: Stage1-DPO + Stage2-DPO

`exp7a-2` is worse than `exp7a-1`.

The current column frequently becomes a full mask-shaped white/pink/purple patch
or a grid-textured object. Many examples are visibly farther from the
`winner/GT` column than DiffuEraser-base. The failure is not subtle: Stage2 DPO
amplifies the patch artifact.

Auxiliary purple/magenta statistics:

- 23 videos reviewed.
- 9 videos have center purple max above 0.05.
- 6 videos have center purple max above 0.10.
- Severe examples overlap with Stage1: `000085`, `000523`, `000365`, `000615`,
  `000467`, `000450`.

DPO diagnostics:

- Final `dpo_loss = 0.0420`.
- Final `implicit_acc = 1.0`.
- Final `lose_gap = 2.5207`.
- Final `mse_l_over_ref_mse_l = 721.9`.
- Last-10 mean `mse_l_over_ref_mse_l = 472.3`.
- Max `mse_l_over_ref_mse_l = 1457`.

Conclusion: Exp7a Stage2 is a strong negative result. The objective is
aggressively separating loser from winner, but visually it often produces a
mask-shaped artifact rather than natural completion.

### Exp8a-1: Stage1-DPO + Stage2-SFT

`exp8a-1` shows the same inpainting-specific failure family on DAVIS, but the
visual form differs from Exp7a.

Instead of large white/pink patches, the current column often has black-purple
or deep-purple blobs, grid/noise texture, dirty boundaries, and unnatural fill
inside the mask. The failure is visible in samples such as `boat`,
`breakdance-flare`, `bmx-bumps`, `car-shadow`, `dance-jump`, `drift-chicane`,
and `dog`.

DPO diagnostics:

- Final `dpo_loss = 0.1649`.
- Final `implicit_acc = 1.0`.
- Final `lose_gap = 1.5160`.
- Final `mse_l_over_ref_mse_l = 300.6`.
- Last-10 mean `mse_l_over_ref_mse_l = 94.6`.

Conclusion: Exp8a Stage1 is not a successful DAVIS inpainting result. It learns
preference separation but often leaves or creates purple/black mask artifacts.

### Exp8a-2: Stage1-DPO + Stage2-DPO

`exp8a-2` does not resolve the Exp8a artifact pattern.

Some examples look close to Exp8a-1, while others continue to show black-purple
patches or boundary artifacts. Stage2 does not produce a consistent visible
improvement over Stage1. DiffuEraser-base is often more conservative and more
natural.

DPO diagnostics:

- Final `dpo_loss = 0.5954`.
- Final `implicit_acc = 1.0`.
- Final `lose_gap = 0.2020`.
- Final `mse_l_over_ref_mse_l = 51.71`.
- Last-10 mean `mse_l_over_ref_mse_l = 31.62`.

Conclusion: Exp8a Stage2 remains negative or inconclusive visually. It does not
turn Exp8a into a reliable inpainting improvement.

### Exp8c-1: GT-win Stage1-DPO + Stage2-SFT

`exp8c-1` is the best mask-inpainting direction among the reviewed Exp8-style
results.

Replacing the winner with GT visibly reduces many of the Exp8a black-purple
artifacts. On the same DAVIS-style samples, `boat`, `car-shadow`, `car-turn`,
`dog`, and `dog-agility` are much closer to background or target content than
in Exp8a. The result is still not fully natural: several samples have patch
boundaries, pasted-looking regions, or imperfect texture alignment. But the
direction is clearly better than Exp8a.

DPO diagnostics:

- Final `dpo_loss = 0.1247`.
- Final `implicit_acc = 1.0`.
- Final `lose_gap = 2.3454`.
- Final `mse_l_over_ref_mse_l = 61.90`.
- Last-10 mean `mse_l_over_ref_mse_l = 38.37`.
- A notable training spike exists: max `mse_w_over_ref_mse_w = 1385`.

Conclusion: GT-win substantially mitigates the purple/black mask artifact
compared with Exp8a, but it does not fully solve realistic inpainting.

### Exp8c-2: GT-win Stage1-DPO + Stage2-DPO

`exp8c-2` does not clearly outperform `exp8c-1`.

Many examples preserve the improvement from GT-win, but several cases reintroduce
black patches or visible pasted regions. Examples include `cows`, `elephant`,
and some drift cases. The Stage2 result is not a clear visual upgrade over the
Stage1 hybrid.

DPO diagnostics:

- Final `dpo_loss = 0.4421`.
- Final `implicit_acc = 1.0`.
- Final `lose_gap = 0.7595`.
- Final `mse_w_over_ref_mse_w = 10.26`.
- Final `mse_l_over_ref_mse_l = 112.6`.
- Last-10 mean `mse_l_over_ref_mse_l = 39.42`.

Conclusion: Exp8c Stage2 keeps part of the GT-win benefit but does not improve
reliability. The final winner/reference ratio is higher than desired, matching
the visual impression that Stage2 can push the model away from conservative
natural completion.

## Cross-Experiment Conclusions

1. Video-generation DPO and partial-mask inpainting DPO should not be interpreted
   as the same phenomenon.

2. `new-exp5` and `new-exp6` show that the regularized DPO setting can improve
   video-generation-style outputs.

3. `exp7a`, `exp8a`, and `exp8c` show that the same DPO family is much less
   reliable for partial-mask inpainting.

4. The dominant mask-inpainting artifact is mask-bound: white/pink/purple
   patches, black-purple blobs, grid texture, and pasted-looking regions.

5. Stage2 DPO is more dangerous than Stage1 DPO in these results. The clearest
   case is Exp7a: Stage2 turns an already imperfect Stage1 into obvious
   mask-shaped patch failures.

6. GT-win helps. Exp8c-1 is visibly better than Exp8a-1 on overlapping DAVIS
   examples. However, GT-win does not fully eliminate patch or boundary artifacts.

7. DPO diagnostics are necessary but not sufficient. High `implicit_acc`, low
   `dpo_loss`, and large `lose_gap` can coexist with visibly worse inpainting.
   For these experiments, diagnostic success means preference separation, not
   guaranteed perceptual inpainting success.

8. The reviewed local evidence supports the following ranking for mask-inpainting
   quality:

   ```text
   Exp8c-1 > Exp8c-2 ~= Exp8a-1/Exp8a-2 > Exp7a-1 > Exp7a-2
   ```

   This ranking is qualitative and based on the reviewed videos in
   `/home/hj/dpo-2-1-exp`.

9. The reviewed local evidence supports the following ranking for
   video-generation-style quality:

   ```text
   new-exp5 >= new-exp6 >> exp5
   ```

10. The central blocker is now clearly localized: DPO can optimize the logged
    preference objective while producing visually invalid mask-region completions.
    This is the primary empirical conclusion from the reviewed videos and
    dpo-diag files.

