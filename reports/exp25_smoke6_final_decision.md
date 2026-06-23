# Exp25 Smoke6 DiffuEraser no-PCM Decision

Date: 2026-06-23

Runtime snapshot:

`/mnt/workspace/hj/nas_hj/runtime_code_snapshots/exp25_934ec73`

Generator:

`diffueraser_or_none_propainter_62d00ca9c76a`

Configuration:

- PCM mode: `none`
- Prior mode: `propainter`
- no-PCM steps: `6`
- no-PCM guidance: `0.0`
- condition: VOR `FG_BG` / `V_obj`
- winner: VOR `BG` / `V_bg`
- mask: VOR object mask
- hard comp: `false`
- VOR-Eval: not used

Technical result:

- DiffuEraser no-PCM smoke6: `6/6` decoded, 24 frames each.
- Output root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/data/external/effecterase_vor/preference_candidates/smoke6_v2_no_pcm/diffueraser/raw_frames`
- Logs:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp25_vor_or_preference_data/diffueraser_no_pcm_smoke_934ec73`

Environment note:

The first no-PCM probe failed because PyAV was missing in the PAI Python
environment. Installing public package `av==17.1.0` fixed the torchvision video
I/O blocker. This was an environment dependency issue, not a DiffuEraser PCM
or checkpoint issue.

Visual review:

I opened all six contact sheets. The model is technically wired correctly: no
black-screen failure, no wrong frame ordering visible in the sheets, no input
copy of the foreground object as a systematic failure, and no hard-compositing
fallback.

Per-sample visual classes:

- `REAL_ENV114_00004_004_02`: `TOO_CLOSE_TO_WINNER`.
- `BLENDER_FOREST039_00117`: `TOO_CLOSE_TO_WINNER`.
- `REAL_ENV024_00002_008_01`: `MEDIUM_HARD_ELIGIBLE`.
- `BLENDER_CON001_00332`: `VALID_BUT_NOT_USEFUL_DARK_TOO_CLOSE`; the scene is
  naturally very dark and the output is too close/easy.
- `REAL_ENV159_00010_003_05`: `HARD_BUT_PLAUSIBLE_WITH_RESIDUAL_ARTIFACT`;
  the object is removed but a visible white phone/edge residual remains.
- `BLENDER_FOREST039_00530`: `VALID_BUT_NOT_USEFUL_TOO_CLOSE`.

Decision:

`DIFFUERASER_NO_PCM_TECHNICAL_PASS`

`READY_GATE128 = false`

Reason:

The strict no-PCM + ProPainter-prior stack is now technically viable on VOR,
but Smoke6 does not yet meet the loser-utility gate. Only one sample is clearly
medium-hard eligible; several are too close to winner, and one contains a
visible residual artifact. Do not start Gate128 until metrics/visual thresholds
are refined and the fixed Smoke6 set shows enough medium-hard signal.

EffectErase remains blocked by missing official assets/wrapper. ProPainter old
smoke remains technical-pass only and still needs the same visual classification
before multimodel expansion.
