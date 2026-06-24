# Exp25 Gate32 Individual Video Reaudit

Status: `INDIVIDUAL_VIDEO_REAUDIT_FRAME_SAMPLE_COMPLETE_PLAYBACK_PENDING`

Run root:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_26_27_followup_20260624/exp25_gate32_individual_video_reaudit_v2`

Scope:

- Existing Gate32 DiffuEraser raw OR candidates only.
- No Gate128 expansion.
- No OR-DPO training.
- No replacement or resampling.

Review method:

- For each of 32 samples, wrote a candidate mp4 from raw frames.
- For each sample, extracted start/middle/end/mask-max/error-max frames.
- For each sample, wrote a contact sheet with mask overlay, raw loser, object/mask crop, affected-region crop, and outside crop.
- Because the current execution channel cannot interactively play mp4 files, every row is marked `VISUAL_REVIEW_PENDING_HEADLESS_FRAME_AUDIT_ONLY` and `reviewer_pass=false`.

Classification:

| class | count |
| --- | ---: |
| medium-hard | 11 |
| trivial-bad | 21 |
| too-close | 0 |
| technical-invalid | 0 |

Key finding:

The prior run-level index over-emphasized purple/black-looking columns because the contact-sheet error-map column is visually dark/purple. The individual audit reads the raw frames directly. `black_frame_ratio=0.0` for all rows, so the main failure is not whole-video black-frame collapse. The dominant failure mode is large mask-region mismatch in the raw loser frames.

Protocol observations:

- `hard_comp=false` and `comp_mode=none`; failures are in raw loser frames, not hard-comp artifacts.
- Most trivial-bad rows have poor mask PSNR but reasonable outside PSNR, so the damage is concentrated in the task region rather than being a global decode failure.
- This audit does not yet distinguish 6-step sampler, mask dilation, ProPainter prior, SFT-48000 domain shift, or official-core effects. That requires the root-cause stack matrix and remains pending.

Outputs:

- CSV: `reports/exp25_gate32_individual_video_reaudit.csv`
- Summary JSON: `reports/exp25_gate32_individual_video_reaudit_summary.json`
- PAI mp4/contact/crop evidence: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_26_27_followup_20260624/exp25_gate32_individual_video_reaudit_v2`

Decision:

Do not mark `VIDEO_REVIEW_PASS`, `LOSER_UTILITY_PASS`, or `DATA_READY`. Do not expand Gate128. Next required step is the DiffuEraser OR root-cause matrix with real alternative inference stacks.
