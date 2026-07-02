# Exp59 Kubric Gate8 Input Materialization

Date: 2026-07-02

Status: `EXP59_KUBRIC_INPUTS_READY`

## Materialized Root

Official input folders were materialized on PAI/NAS at:

`/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp59_void_kubric_gate8_inference/official_inputs`

Each sample folder contains:

```text
input_video.mp4 -> Exp58B rgb_full.mp4
quadmask_0.mp4 -> Exp58B mask.mp4
rgb_removed.mp4 -> Exp58B rgb_removed.mp4, for metrics only
metadata.json -> Exp58B metadata.json
prompt.json
```

The official loader only needs `input_video.mp4`, `quadmask_0.mp4`, and `prompt.json`; `rgb_removed.mp4` is included for later metric/review code.

## Prompt Rule

Prompts were generated from the Kubric metadata `background` field and only describe the remaining background scene. They do not describe removal. Example:

```json
{"bg": "A clean abandoned games room 02 scene with the remaining objects."}
```

## Validation

- Sample folders: 8
- `input_video.mp4` decode: 8/8
- `quadmask_0.mp4` decode: 8/8
- `rgb_removed.mp4` decode: 8/8
- Frame-count match across input/mask/GT: yes
- Mask values preserved as `0|63|127|255`: yes
- VOR paths mixed in: no
- `target_hit=true`: 0/8
- `target_hit=false`: 8/8

## Storage Caveat

The requested experiment output root under:

`/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp59_void_kubric_gate8_inference`

is not writable by `hj` on PAI. The PAI log/runtime roots are writable:

- `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp59_void_kubric_gate8_inference`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp59_void_kubric_gate8_inference`

Official inference outputs should therefore be saved under the writable log/runtime tree and this fallback must be recorded in all runtime reports.

## Evidence

- Manifest: `manifests/exp59_kubric_gate8_inference_inputs.jsonl`
- CSV: `reports/exp59_kubric_input_materialization.csv`
- Summary: `reports/exp59_kubric_input_materialization_summary.json`

## Safety

- Training run: no
- Preference forward: no
- Zero-gap: no
- One-step: no
- 10-step: no
- Official VOID source modified: no
- `inference/metrics.py` modified: no
- Shared trainer modified: no
