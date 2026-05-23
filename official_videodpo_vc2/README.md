# official_videodpo_vc2

Purpose: protect and document the completed official-VideoDPO VC2 baseline/DPO experiment, including qualitative and full VBench evaluation.

## Scope

Changed in the experiment:

- Ran official VideoDPO VC2 DPO fine-tuning.
- Evaluated full VBench.
- Generated 30 SBS qualitative videos comparing reproduced base and official VideoDPO.

Not changed:

- VC2 model architecture.
- Official VideoDPO training objective.
- VBench metric definitions.

## Data / Mask / Preference Definition

- Win source: original VideoDPO winner.
- Loser source: original VideoDPO rejected video.
- Mask: not a DiffuEraser inpainting mask experiment.
- Training task: official VideoDPO VC2.
- Offline/online loser generation: not applicable.

## Existing Entry Points

- `DPO_finetune/scripts/pai_videodpo_vc2_official_repro.sh`
- Official repo path is configured by `OFFICIAL_VIDEODPO_ROOT`.
- VC2 weights should be configured by `VC2_WEIGHT_ROOT`.

## Recorded PAI Artifacts

These paths were recorded in PRDs but are not mounted in the current `hal-9000` audit session:

- `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/videodpo_vc2_dpo_official_clean/pai-vc2-dpo-official-full-gpu0-3-gb8-step3000-20260521_061414/checkpoints/last.ckpt`
- `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/vbench_full/pai-vc2-official-step3000-full-vbench-20260521_141824`
- `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/logs/qual_sbs_30/vc2_and_diffueraser_20260522/vc2_base_vs_official_videodpo_samplemix`

## Recorded Score

| Model | Total | Quality | Semantic | MeanRaw |
| --- | ---: | ---: | ---: | ---: |
| Official VC2 VideoDPO step3000 | 80.5997 | 82.8055 | 71.7763 | 0.6596 |
