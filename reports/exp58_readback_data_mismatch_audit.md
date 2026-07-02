# Exp58 Readback Data Mismatch Audit

Status: `EXP58_READBACK_DONE`

## Answers

1. What data was used in failed VOID adapter attempts?

Exp50-Exp57 used VOR-Train-derived train4 / heldout4 plus Q0/Q1/Q2/Q3 quadmask variants. Losers were same-model VOID outputs where usable or controlled local corruptions. VOR-Eval was excluded.

2. How does VOR-derived data differ from official VOID data?

VOR-derived data starts from object-removal foreground/background pairs and constructs affected/quadmask regions from frame differences. Official VOID data is generated paired counterfactual data: HUMOTO human-object physics removal or Kubric object-only physics counterfactuals with explicit full/removed/altered physics passes.

3. Does official VOID release prebuilt training data?

The repo checkout includes `datasets/void_train_data.json` with 6383 metadata rows, but not the rendered `training_data/` video folders themselves. The README expects generated or externally available `training_data/`.

4. What official data can we generate now?

Potentially the Kubric object-only counterfactual pipeline, because it can fetch public KuBasic/GSO/HDRI manifests. HUMOTO is not currently feasible without manual HUMOTO, Mixamo, and Blender asset preparation.

5. Is Kubric feasible without external dataset download?

Maybe, but only if Kubric, PyBullet, Blender/bpy, ffmpeg, and public GCS assets are reachable in the isolated environment. Exp51 previously found Kubric/PyBullet/Blender missing.

6. Is HUMOTO feasible now or blocked by access / Blender / Mixamo?

Blocked. HUMOTO requires license agreement and manual download; Mixamo Remy/Sophie FBX assets require a free Adobe/Mixamo account and are absent. Blender is also required.

7. What does this experiment need to prove?

It must determine whether VOID-native Kubric data reduces the transition-region regressions seen on VOR-derived data, or whether the wrapper/objective remains negative even on native generated data.

8. Why no 10-step yet?

No Exp50-Exp57 one-step rescue produced a PASS under visual and quadmask-aware gates. Exp58 is a data diagnostic and is one-step only.

## Official VOID Data Notes

Official `data_generation/README.md` says HUMOTO generation produces `rgb_full.mp4`, `rgb_removed.mp4`, `mask.mp4`, and `metadata.json`. The Kubric pipeline produces `rgb_full.mp4`, `rgb_removed_objects_invisible.mp4`, `rgb_altered_physics.mp4`, `mask.mp4`, and `metadata.json`; the mask values are 0, 63, 127, and 255.

## Decision

Proceed to isolated Kubric environment smoke. Do not generate fake native data if Kubric/Blender/assets fail.
