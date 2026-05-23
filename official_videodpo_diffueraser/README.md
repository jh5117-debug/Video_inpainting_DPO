# official_videodpo_diffueraser

Purpose: protect and document the completed official-VideoDPO DiffuEraser model-ablation experiment.

This directory contains the minimal adapter that lets the official VideoDPO Lightning trainer host DiffuEraser. It should remain stable because the May 21-22 VC2 -> DiffuEraser experiment depends on it.

## Scope

Changed in the experiment:

- Replaced the official VideoDPO/VC2 model path with a DiffuEraser bridge.
- Kept the official VideoDPO training skeleton.
- Used full-mask DiffuEraser batch contract.
- Ran stage1 and stage2 fine-tuning, then fullmask full VBench.

Not changed:

- DPO loss definition.
- Dataset preference semantics.
- Metric definitions.
- Full-mask bridge training contract.

## Data / Mask / Preference Definition

- Win source: original VideoDPO winner.
- Loser source: original VideoDPO rejected video.
- Mask for training: full mask / black condition.
- Partial mask: not used.
- Comp: not used.
- Changed variable: model backbone/adapter, not data.

## Existing Code

- `official_videodpo_diffueraser/data.py`
- `official_videodpo_diffueraser/models.py`
- `DPO_finetune/configs/official_diffueraser_stage1.yaml`
- `DPO_finetune/configs/official_diffueraser_stage2.yaml`
- `DPO_finetune/scripts/pai_official_diffueraser_stage.sh`
- `tools/generate_diffueraser_fullmask_vbench.py`
- `DPO_finetune/scripts/h20_diffueraser_fullmask_vbench.sh`

## Recorded PAI Artifacts

These paths were recorded in PRDs but are not mounted in the current `hal-9000` audit session:

- `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/official_diffueraser_stage1/pai-official-diffueraser-stage1-full-gpu4-7-gb8-step3000-20260521_072559/last_weights`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/official_diffueraser_stage2/pai-official-diffueraser-stage2-full-gpu4-7-gb8-step3000-20260521_150540/last_weights`
- `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/logs/vbench_fullmask/pai-official-diffueraser-stage2-vs-base-full-vbench-20260522_002926`

## Recorded VBench

| Model | Total | Quality | Semantic | MeanRaw |
| --- | ---: | ---: | ---: | ---: |
| DiffuEraser-Base-Fullmask | 64.6162 | 74.4651 | 25.2204 | 0.3935 |
| DiffuEraser-Stage2-Fullmask | 73.6463 | 78.4804 | 54.3099 | 0.5560 |
| Delta | +9.0301 | +4.0153 | +29.0894 | +0.1625 |
