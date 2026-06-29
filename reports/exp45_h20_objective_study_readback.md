# Exp45 H20 Objective Study Readback

Generated: `2026-06-29T16:01:55,415804093+08:00` on `instance-afs92r3e`.

## Git State

- Worktree: `/home/nvme01/H20_Video_inpainting_DPO_exp45_objective_study`
- Branch: `research/exp45-h20-diffueraser-videopainter-objective-study-20260629`
- HEAD: `17b99a421fbf1bb79a446713a1f30ef6c8ecc769`
- Base selected: `origin/research/exp27-paper-grounded-preference-study` (`17b99a421fbf1bb79a446713a1f30ef6c8ecc769`)
- Base fallback: not used; Exp27 remote exists and is the requested first-priority base.
- Exp31 reference HEAD: `0fc89f0f50e57206e0b3f9ee8b8cfa0e08a20a26`
- origin/main HEAD: `34844d75aba585542b311098417f67c7274f6434`
- Plain `git status --short`: `clean`
- `git status --short --ignore-submodules=all`: `(clean with --ignore-submodules=all)`
- Pre-write `git diff --stat`: `(no diff before Exp45 writes)`
- Pre-write `git diff --check`: `(no diff-check issues before Exp45 writes)`

```text
17b99a4 Run Exp27 Linear-DPO true-model micro gate
265bfe0 Record Exp27 true-model SDPO gate
e708715 Add Exp27 true-model tiny-step gate
4aed0d8 Add Exp27 true model objective parity gate
005e603 Record final PAI permission recovery
2b1a491 Record Exp27 PAI postmaintenance blockers
f73e542 Scan Exp27 SDPO real residual distribution
3d95c9e Record Exp27 PAI persistence completion
3183aec Record Exp27 PAI persistence blocker
1482183 Read back Exp27 true model forward status
1a2afd6 Add Exp27 distribution scan and LocalDPO path fix
8c4ed57 Record Exp27 nontrivial parity gates
```

## Source Material Read

| Material | Ref | Status |
| --- | --- | --- |
| Exp27 PRD | `HEAD:PRD/49_exp27_paper_grounded_preference_study.md` | present |
| Exp27 Linear true model report | `HEAD:reports/exp27_linear_true_model_10step.md` | present |
| Exp31 PRD | `origin/research/exp31-videopainter-2000step-longrun-20260627:PRD/49_exp31_videopainter_2000step_longrun.md` | present |
| Exp31 final decision | `origin/research/exp31-videopainter-2000step-longrun-20260627:reports/exp31_vp_2000_final_decision.md` | present |
| Exp31 LPIPS/Ewarp | `origin/research/exp31-videopainter-2000step-longrun-20260627:reports/exp31_vp_2000_lpips_ewarp_metrics.md` | present |
| Exp31 paper evidence | `origin/research/exp31-videopainter-2000step-longrun-20260627:reports/exp31_vp_2000_paper_evidence.md` | present |
| Exp26 PRD | `origin/research/exp26-videopainter-dpo-v2:PRD/48_exp26_videopainter_dpo_v2.md` | present |
| Exp26 Step50 final | `origin/research/exp26-videopainter-dpo-v2:reports/exp26_vp_50step_final.md` | present |
| Exp26 shadow final | `origin/research/exp26-videopainter-dpo-v2:reports/exp26_vp_shadowdev_final_decision.md` | present |
| Exp39 H20 mirror PRD | `origin/research/exp39-h20-minimax-mirror-bf16-20260628:PRD/55_exp39_h20_minimax_mirror_bf16.md` | present |
| Exp41 protocol report | `origin/research/exp41-h20-minimax-parallel-bf16-20260629:reports/exp41_h20_minimax_official_protocol_audit.md` | present |
| Exp43 PRD | `origin/research/exp43-h20-minimax-stage2-sft-runner-20260629:PRD/57_exp43_h20_minimax_stage2_sft_runner.md` | present |
| Exp38 failure taxonomy | `origin/research/exp38-minimax-full-adapter-breakthrough-20260628:reports/exp38_minimax_failure_taxonomy.md` | present |
| Exp42 mining | `origin/research/exp42-pai-minimax-successful-removal-badnoise-20260629:reports/exp42_minimax_official_successful_removal_mining.md` | present |

Notes:
- The Exp45 worktree itself is based on Exp27, so Exp31/26/39/41/43/42 material was read from remote refs with `git show` instead of merging those branches.
- The main repo worktree has a broken `external/VBench` submodule reference that makes plain `git status --short` fail; Exp45 uses `--ignore-submodules=all` for status until that unrelated submodule metadata is repaired elsewhere.

## H20 GPU Read-Only Audit

GPU summary from `nvidia-smi`:

```text
0, NVIDIA H20, 50119, 97871, 27
1, NVIDIA H20, 50249, 97871, 23
2, NVIDIA H20, 49867, 97871, 5
3, NVIDIA H20, 49500, 97871, 7
4, NVIDIA H20, 50445, 97871, 25
5, NVIDIA H20, 34907, 97871, 0
6, NVIDIA H20, 50299, 97871, 22
7, NVIDIA H20, 50287, 97871, 13
```

Compute apps:

```text
GPU-53e27608-e06c-4088-85fd-81412f1f451d, 2435051, python, 25232
GPU-53e27608-e06c-4088-85fd-81412f1f451d, 2437100, python, 24824
GPU-a799bf88-f766-89c7-3edb-5ece7dd513e6, 2435205, python, 25232
GPU-a799bf88-f766-89c7-3edb-5ece7dd513e6, 2436876, python, 24982
GPU-86044016-7665-effd-e318-ec67c4e61190, 2435394, python, 24824
GPU-86044016-7665-effd-e318-ec67c4e61190, 3810237, python, 25008
GPU-0b7c9457-c09b-532c-07aa-fe3ee306411d, 2435416, python, 24924
GPU-0b7c9457-c09b-532c-07aa-fe3ee306411d, 2687167, python, 24550
GPU-18eab895-41e6-3062-a5df-b104db5e2cd0, 2435885, python, 25402
GPU-18eab895-41e6-3062-a5df-b104db5e2cd0, 3168569, python, 25008
GPU-c15fc8f0-9d89-86bb-701d-38f848f9366e, 2438246, python, 25008
GPU-c15fc8f0-9d89-86bb-701d-38f848f9366e, 258439, python, 9874
GPU-ca37d462-7bfc-213b-e46e-bef520750458, 2436226, python, 25256
GPU-ca37d462-7bfc-213b-e46e-bef520750458, 2922549, python, 25008
GPU-40d1090a-9ef2-98c0-7f6d-48c83ab04348, 2436243, python, 25162
GPU-40d1090a-9ef2-98c0-7f6d-48c83ab04348, 3812181, python, 25090
```

Process classification: all active compute PIDs observed during readback are root-owned LIBERO evaluation commands (`experiments/libero/eval_libero_single.py`), not Video_inpainting_DPO. They are treated as unknown/external H20 jobs and were not signaled, paused, killed, or reniced.

## H20 Asset Snapshot

| Asset | Path | Status | Size |
| --- | --- | --- | --- |
| h20_data | `/home/nvme01/H20_Video_inpainting_DPO/data` | present | 125G |
| h20_mirror | `/home/nvme01/H20_Video_inpainting_DPO/data/h20_mirror` | present | 14G |
| exp45_mirror | `/home/nvme01/H20_Video_inpainting_DPO/data/h20_mirror/exp45_objective_study` | missing | n/a |
| diffueraser_weights | `/home/nvme01/H20_Video_inpainting_DPO/weights/diffuEraser/converted_weights_step48000` | present | 8.2G |
| diffueraser_third_party_weights | `/home/nvme01/H20_Video_inpainting_DPO/third_party_video_inpainting/weights/diffueraser/converted_weights_step48000` | present | 8.2G |
| videopainter_reports_in_base | `/home/nvme01/H20_Video_inpainting_DPO_exp45_objective_study/reports/exp31_vp_2000_final_decision.md` | missing | n/a |

Interpretation:
- H20 already has DiffuEraser source and Step48000-style weight directories locally.
- H20 has the MiniMax mirror from prior work, but MiniMax is out of scope for this Exp45 run.
- The dedicated Exp45 mirror root is not created yet; asset inventory must precede transfer.
- VideoPainter Exp31/Exp26 evidence exists in git reports, but runtime assets/checkpoints must still be inventoried before any transfer or smoke.

## Required Answers

1. Which branch is used as base?

`origin/research/exp27-paper-grounded-preference-study` at `17b99a421fbf1bb79a446713a1f30ef6c8ecc769`. This is the requested first-priority base; no fallback was used.

2. Which DiffuEraser objective-study code exists?

Exp27 contains paper-grounded primitive/parity work for SDPO, Linear-DPO Frozen/EMA, and LocalDPO compatibility. The PRD records true-model SDPO completion and true-model Linear 1/10-step completion, while the full O0-O6 objective study remains pending. Exp45 will add only isolated code under `exp45_h20_objective_study/`.

3. Which VideoPainter objective-study code exists?

VideoPainter has Exp26 Step50 micro evidence and Exp31 Step2000 long-run evidence. Exp31 final status is `VIDEOPAINTER_2000_POSITIVE`; this is cross-backbone evidence, not a reason to rerun all O0-O6 on VideoPainter. Exp45 should transfer at most selected top methods after DiffuEraser completes.

4. Which H20 assets are already present?

H20 has local project data, DiffuEraser code/weights directories, and the previous MiniMax mirror. The dedicated Exp45 asset mirror is absent. Search/shadow/train manifests and exact checkpoint identities must be inventoried before transfer.

5. Which PAI assets must be pulled?

Only the minimal files needed for DiffuEraser objective training/evaluation and selected VideoPainter transfer: manifests, referenced winner/loser/condition/mask videos or latents, required checkpoints/weights, metric assets, and prior evidence reports. Full VOR archive, unrelated cache, old failed outputs, MiniMax large outputs, and EffectErase datasets must not be transferred.

6. Which experiments must not be repeated?

Do not repeat Exp26 Step50, Exp31 Step2000, Exp39/41/43 MiniMax plumbing/SFT ladder, Exp42/44 MiniMax PAI mining, or any VOR-Eval training/selection. Exp45 is a new isolated objective study.

7. What is LoVI-DPO main method?

LoVI-DPO is the main localized video inpainting DPO method: GT/clean winner, loser handling, reference-normalized DPO-style gap, winner anchoring, and mask/outer-boundary localized weights with outside preservation. External methods are not claimed as original contributions.

8. What are Linear-DPO / SDPO / LocalDPO roles?

They are external baselines, objective ablations, and integration studies. SDPO tests safe-lambda diffusion preference geometry, Linear-DPO tests linear utility with frozen/EMA reference variants, and LocalDPO-style tests a 24F local-corruption adaptation rather than an official LocalDPO result.

9. Why DiffuEraser first and VideoPainter second?

DiffuEraser is the primary successful adapter platform and is suitable for complete O0-O6 objective ablation. VideoPainter already has positive long-run evidence, so it should only receive selected transfer methods after DiffuEraser identifies a real top candidate.

10. What is the exact gate for promotion?

A method can be promoted only after preregistered metrics, locked search/shadow protocol, strict checkpoint reload, no systematic winner degradation, safe LPIPS/Ewarp/outside behavior, and full video review. Metrics-only promotion is forbidden.

## Current Status

- Asset status: `H20_OBJECTIVE_STUDY_ASSETS_PARTIAL` until inventory and checksums complete.
- Env status: `H20_OBJECTIVE_STUDY_ENV_BLOCKED` for GPU smoke/training while external H20 GPU tasks occupy all GPUs; CPU/Git work can continue.
- DiffuEraser status: `DIFFUSERASER_OBJECTIVE_STUDY_BLOCKED` until assets and environment are ready.
- VideoPainter status: `VIDEOPAINTER_TRANSFER_BLOCKED` until DiffuEraser selects a top method.
- Paper status: `OBJECTIVE_STUDY_INCONCLUSIVE`.

Forbidden statuses not asserted: `UNIVERSAL_ADAPTER`, `ALL_MODELS_SUPPORTED`, `FINAL_SOTA`, `TOP_CONFERENCE_NOVELTY_CONFIRMED`.
