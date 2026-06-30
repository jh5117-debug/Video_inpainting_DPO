# Exp50 VOID VOR Gate8 Official Inference Smoke

Status: VOID_INFERENCE_SMOKE_PASS

## Protocol

- Environment: `/home/hj/conda_envs/void_exp50_official_v2`
- Official repo: `/mnt/nas/hj/H20_Video_inpainting_DPO/third_party/VOID/Netflix_void-model`
- Script: `inference/cogvideox_fun/predict_v2v.py`
- Config: `config/quadmask_cogvideox.py`
- Weights: `void_pass1.safetensors`
- Dataset: existing Exp50 VOR-Train Gate8 only; VOR-Eval excluded.
- Runtime compatibility changes: prompt-schema view added `bg` from original prompt, `input_video.mp4` symlinked to `rgb_full.mp4`, and PATH used the isolated-env static ffmpeg from `imageio_ffmpeg` because system ffmpeg on PAI cannot load `libblas.so.3`.
- No hard comp, no training, no optimizer step, no VOID source modification.

## Gate Result

- Technical valid: 8 / 8
- Usable or bounded loser: 6 / 8
- Classification counts: {'TOO_CLOSE': 2, 'VOID_OUTPUT_USABLE': 4, 'MEDIUM_HARD_LOSER': 2}
- Systematic outside collapse: no
- LPIPS/Ewarp/TC: unavailable in this smoke; not used for gate promotion.

## Visual Review

| sample | type | classification | PSNR | outside PSNR | notes |
|---|---|---:|---:|---:|---|
| BLENDER_CON001_00742 | BLENDER | TOO_CLOSE | 37.27 | 37.36 | technical valid and outside safe, but visual target delta is tiny; not a useful loser or adapter signal. |
| BLENDER_CON001_00843 | BLENDER | VOID_OUTPUT_USABLE | 34.05 | 36.23 | technical valid; local removal follows target with bounded residual; outside remains stable. |
| BLENDER_CON001_00636 | BLENDER | MEDIUM_HARD_LOSER | 30.30 | 34.79 | technical valid; local area remains imperfect versus target while outside is safe, useful as bounded loser signal. |
| BLENDER_CON001_00744 | BLENDER | TOO_CLOSE | 36.42 | 37.17 | technical valid and outside safe, but object/affected change is too small for preference mining. |
| REAL_ENV200_00001_006_02 | REAL | VOID_OUTPUT_USABLE | 27.70 | 28.57 | technical valid; small foreground object removed with no global collapse, but signal is mild. |
| REAL_ENV219_00001_003_05 | REAL | MEDIUM_HARD_LOSER | 19.64 | 25.93 | technical valid; person is removed but a visible localized haze/residual remains near the edited region; outside mostly sane. |
| REAL_ENV259_00102_002_04 | REAL | VOID_OUTPUT_USABLE | 21.20 | 29.00 | technical valid; lower-body foreground removed with plausible background and bounded boundary artifacts. |
| REAL_ENV102_00001_002_02 | REAL | VOID_OUTPUT_USABLE | 34.81 | 35.82 | technical valid; clean person removal, stable background, no visible outside collapse in evidence frames. |

## Interpretation

VOID pass1 is technically runnable on PAI for the VOR-Train Gate8 conversion. The outputs are not a training or adapter-positive claim: the smoke only shows that official VOID inference can generate technically valid raw outputs, including several usable removals and two bounded medium-hard loser candidates. The next allowed step is G0 micro-data preparation; optimizer steps remain locked behind G1.
