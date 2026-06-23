# PRD 48: Exp26 VideoPainter DPO v2

## Motivation

Exp14 showed that VideoPainter can run through an isolated DPO adapter trainer,
but the 2000-step gate underperformed the official VideoPainter baseline and
showed saturation / loser-dominant risks. Exp26 restarts VideoPainter as a v2
track with a stricter adapter ladder instead of directly running long training.

## Required Ladder

All new backbones, including VideoPainter v2, must pass:

| Level | Gate |
| --- | --- |
| L0 | Official baseline strict-load and native inference reproduction |
| L1 | Native loss parity with the official noising/target path |
| L2 | Policy=reference DPO zero-gap test, DPO loss about log(2) |
| L3 | 1-step update, strict checkpoint save/reload, output changes |
| L4 | 10-step smoke on real preference data |
| L5 | Micro gates at 50/100/250/500 with fixed search-dev eval |
| L6 | Promotion to 1000/1500/2000 only if checkpoint curve improves |
| L7 | Final evaluation only after checkpoint is locked |

## Initial v2 Fixes

The first Exp26 commit copies Exp14 into `exp26_videopainter_dpo_v2/` and fixes
several issues before any GPU training:

- optimizer construction now exposes AdamW betas, epsilon, weight decay, and LR
  instead of a hard-coded `AdamW(..., lr=...)`;
- `noised_image_dropout` is applied to first-frame image latents;
- first-frame GT conditioning now forces winner, loser, condition, and mask to
  be mutually consistent;
- formal frame count defaults to 49 frames and requires the loader to select
  exactly 49 frames; 16-frame inputs now fail instead of silently trimming to
  13;
- 13-frame runs require `--plumbing_only_13f` and are labelled plumbing-only;
- `--first_frame_gt` / `--no-first_frame_gt` now controls whether first-frame
  winner/loser/condition/mask consistency is enforced;
- official optimizer/scheduler defaults can be parsed from the current
  VideoPainter trainer and written to a locked JSON before parity gates;
- `itertools.cycle(loader)` is removed in favor of a resumable epoch iterator;
- loser-dominant diagnostics use the project definition: correct preference
  and loser degradation greater than winner improvement;
- strict state-dict reload helper is present for checkpoint identity tests.

## Current Status

Status: `VP2_STATIC_FIXES_STARTED`

No VideoPainter v2 GPU training, self-loser generation, DAVIS50 evaluation, or
long checkpoint gate has been started in this commit.
