# Exp44 PAI MiniMax Targeted Same-Source Mining

Status: `MINIMAX_STAGE2_DATA_HANDOFF_PARTIAL`

## Purpose

Exp44 addresses MiniMax's current data bottleneck: Exp42 found row-level successful-removal and failure signals, but only `7` same-source success/failure overlap groups. Exp44 performs targeted second-pass mining and visual relabeling to construct clean same-source pairs for bad-noise v4 and Stage2-style H20 handoff.

## Branch and Roots

- Branch: `research/exp44-pai-minimax-targeted-same-source-mining-20260629`
- Base: `origin/research/exp42-pai-minimax-successful-removal-badnoise-20260629`
- HAL worktree: `/home/hj/H20_Video_inpainting_DPO_exp44_pai_minimax_targeted`
- Requested PAI code root: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp44_pai_minimax_targeted`
- Output root: `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp44_pai_minimax_targeted_same_source_mining`
- Log root: `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp44_pai_minimax_targeted_same_source_mining`
- Runtime root: `/mnt/nas/hj/H20_Video_inpainting_DPO/runtime/exp44_pai_minimax_targeted_same_source_mining`

## Boundaries

- No H20 worktree/output/GPU modification.
- No training by default; if data gates pass, only dataloader / one-batch forward smoke is allowed.
- No optimizer step in this prompt.
- No VOR-Eval training/selection/tuning.
- No hard comp.
- No modifications to `inference/metrics.py`, shared trainers, MiniMax official repo source, or Exp1-Exp43 historical outputs.
- No MiniMax third-backbone-positive, universal-adapter, final-SOTA, or top-conference novelty claims.

## 2026-06-29 Readback and Source Plan

Status: `EXP44_TARGETED_READBACK_COMPLETED`.

Exp42 verified inputs:

- success rows: `52`
- failure rows: `80`
- success scene groups: `18`
- failure scene groups: `29`
- same-source overlap groups: `7`

Plan:

- Group C overlap groups: `7`
- Group A success-only groups: `11`
- Group B failure-only groups: `22`
- Group D fallback groups: `16`

Reports:

- `reports/exp44_minimax_targeted_readback.md`
- `reports/exp44_source_group_plan.csv`
- `reports/exp44_source_group_plan.json`

Next status target: `MINIMAX_TARGETED_MINING_COMPLETED`.

PAI GPU readback: GPU0/GPU1 were occupied by unrelated `qxq_sample_valtest_v0.py`
root-owned jobs during Milestone A readback. They were not old MiniMax project
processes, so Exp44 sent no signals and launched no GPU task.

## 2026-06-29 Targeted Source Manifest and Runner Preparation

Status: `EXP44_TARGETED_SOURCE_MANIFEST_READY`.

Milestone B preparation added Exp44-isolated helper scripts and built the locked
targeted source/seed manifest:

- source rows: `40` from A/B/C groups only;
- fallback groups included: `false`;
- initial candidate budget: `452`;
- missing source rows: `0`;
- manifest SHA256: `5147839e1e2d60e0ecc9c77a438a934918605b5fa550fa58d1e3291df7be168b`.

New helpers:

- `exp44_pai_minimax_targeted_same_source_mining/scripts/build_targeted_mining_manifest.py`
- `exp44_pai_minimax_targeted_same_source_mining/scripts/mine_targeted_candidates.py`

The runner reuses the already audited MiniMax official pipeline and Exp30
`run_pipeline` helper. It preserves the Exp44 protocol: raw output primary,
UniPCMultistepScheduler, `num_inference_steps=12`, `iterations=6`, float16,
no VOR-Eval, no hard comp, no training, and no optimizer step.

Reports/manifests:

- `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_targeted_source_manifest.jsonl`
- `reports/exp44_targeted_source_manifest_summary.json`

GPU mining is still pending actual GPU0/GPU1 availability.

## 2026-06-29 Targeted Second-Pass Mining Completed

Status: `MINIMAX_TARGETED_MINING_COMPLETED`.

PAI GPU0/GPU1 were released by external/manual cleanup, then Exp44 launched
two task-parallel official MiniMax workers:

- GPU0 worker PID/PGID: `2263394` / `2263394`;
- GPU1 worker PID/PGID: `2266642` / `2266642`;
- watcher PID: `2257670`;
- monitor PID after live-count fix: `2279260`;
- output root:
  `/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp44_pai_minimax_targeted_same_source_mining/targeted_mining_20260629_161742`.

Protocol:

- `UniPCMultistepScheduler`;
- `num_inference_steps=12`;
- `iterations=6`;
- `float16`;
- no CFG;
- raw output primary;
- no hard comp;
- no VOR-Eval;
- no training;
- no optimizer step.

Mining results:

- total candidates: `452` / `452`;
- technical failed candidates: `0`;
- automatic successful-removal candidates: `138`;
- automatic medium-hard failure candidates: `231`;
- automatic boundary-bad candidates: `31`;
- automatic fogging/over-erasure candidates: `25`;
- automatic too-close candidates: `27`;
- automatic same-source pair capacity: `26`;
- automatic overlap groups: `13`;
- all-candidate manifest SHA256:
  `fd4152b5e789b2d22ae11bd83b2cbbe69eee51347a1444d04178c810f957fbd3`;
- auto-success manifest SHA256:
  `b0ebd76f442425e687ed42af50ce36dcda941526dded21246d9717e03ab88e6b`;
- auto-failure manifest SHA256:
  `0b01156fb2becbe9d7cf9f6086a4f3a74fd66ecf37bff31d2b54019f45389a24`.

Important caveat: these are automatic labels only. Exp44 has crossed the
minimum auto-level same-source candidate line (`26 >= 24`), but the formal
same-source pair gate is still locked until strict visual relabeling rejects
borderline, fogging, outside-damaged, too-close, and technical-invalid rows.

Reports/manifests:

- `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_targeted_candidates_all.jsonl`
- `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_targeted_success_auto.jsonl`
- `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_targeted_failure_auto.jsonl`
- `reports/exp44_targeted_mining.md`
- `reports/exp44_targeted_mining_metrics.csv`
- `reports/exp44_targeted_mining_group_yield.csv`
- `reports/exp44_targeted_mining_summary.json`

Next status target: `MINIMAX_TARGETED_RELABEL_COMPLETED`.

## 2026-06-29 Strict Visual Relabeling Completed

Status: `MINIMAX_TARGETED_RELABEL_COMPLETED`.

Codex opened and inspected all `47` selected candidate review pages generated
from the targeted mining output. The relabel pass is intentionally conservative:
fogging, over-erasure, too-close rows, boundary destruction, outside damage, and
metric-only failures without visible usefulness were kept out of the success and
medium-hard pools.

Relabel counts:

- total candidates: `452`;
- selected page rows reviewed: `369`;
- `SUCCESS_CLEAN`: `33`;
- `SUCCESS_USABLE`: `92`;
- usable success including clean: `125`;
- `FAILURE_MEDIUM_HARD`: `137`;
- rejected / borderline / non-usable: `190`;
- same-source groups with both usable success and medium-hard failure: `10`;
- one-to-one same-source pair precheck: `18`;
- capped same-source combination precheck: `40`.

Important interpretation: Milestone C completes label purification only. It
does not unlock training and it does not claim MiniMax quality improvement.
Milestone D must still construct explicit same-source pairs, enforce
scene-disjoint train/search/shadow splits, and verify the formal `>=24` usable
pair gate before bad-noise v4 or any Stage2 handoff is trusted.

Reports/manifests:

- `reports/exp44_targeted_visual_relabel.md`
- `reports/exp44_targeted_visual_relabel.csv`
- `reports/exp44_targeted_visual_relabel_group_yield.csv`
- `reports/exp44_targeted_visual_relabel_summary.json`
- `reports/exp44_targeted_visual_review_pages/`
- `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_success_clean.jsonl`
- `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_success_usable.jsonl`
- `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_failure_medium_hard.jsonl`
- `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_rejected_borderline.jsonl`
- `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_targeted_visual_relabel_all.jsonl`

Next status target: `MINIMAX_SAME_SOURCE_PAIR_GATE_PASSED` or
`MINIMAX_SAME_SOURCE_PAIR_YIELD_INSUFFICIENT`.

## 2026-06-29 Same-Source Pair Gate Passed

Status: `MINIMAX_SAME_SOURCE_PAIR_GATE_PASSED`.

Milestone D constructed clean MiniMax-native same-source success/failure pairs
from the visually relabeled pools. Pairing is strictly within source group;
there are no cross-source pairs.

Pair construction:

- total usable same-source pairs: `40`;
- minimum gate: `24`;
- target: `48`;
- train/search/shadow candidate split: `24` / `8` / `8`;
- split scene-group overlap: `0`;
- max pairs per group: `4`;
- source groups with pairs: `10`;
- winner for DPO preference fields: GT background `V_bg`;
- pseudo-success retained only as Stage2 distillation target metadata;
- loser: same-source MiniMax raw output labeled `FAILURE_MEDIUM_HARD`.

This passes the Exp44 minimum same-source pair gate and unlocks Milestone E
bad-noise v4 state construction. It still does not unlock SFT/DPO training or
MiniMax quality-positive language. No training, optimizer step, VOR-Eval use,
hard comp, or H20 modification occurred.

Reports/manifests:

- `reports/exp44_same_source_pair_construction.md`
- `reports/exp44_same_source_pair_construction.csv`
- `reports/exp44_same_source_pair_group_yield.csv`
- `reports/exp44_same_source_pair_summary.json`
- `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_same_source_pairs_all.jsonl`
- `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_same_source_pairs_train_candidates.jsonl`
- `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_same_source_pairs_search_candidates.jsonl`
- `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_same_source_pairs_shadow_candidates.jsonl`

Next status target: `MINIMAX_BADNOISE_V4_READY` or
`MINIMAX_BADNOISE_V4_BLOCKED`.

## 2026-06-29 Bad-Noise v4 State Construction

Status: `MINIMAX_BADNOISE_V4_READY`.

Milestone E constructed MiniMax-native bad-noise v4 state records from the
same-source pair set. The construction used CPU-side residual and gradient
proxy metrics only; it did not run backpropagation, training, optimizer steps,
or any model update.

State construction:

- total state records: `40`;
- usable H-state records: `26`;
- minimum usable H-state gate: `24`;
- H1 local-failure-hard states: `20`;
- H3 winner-safe states: `20`;
- hard-state local/random gradient-proxy ratio mean: `2.676106`;
- hard-state local/random gradient-proxy ratio median: `2.280567`;
- random local/outside median: `5.144638`;
- outside-risk median: `0.342387`;
- manifest SHA256:
  `89f26f2a3c2a2f8f9f09ae14d0d15d5fa38a73dccb4a345d0ee56123f09c1d62`.

Interpretation:

- Exp44 now has enough same-source MiniMax failure states to hand off a
  Stage2-style SFT/DPO data package;
- `gradient_proxy_norm` is a residual-based proxy, not a true autograd
  gradient;
- no training, optimizer step, VOR-Eval use, hard comp, H20 modification, or
  MiniMax quality-positive claim occurred.

Reports/manifests:

- `reports/exp44_badnoise_v4_states.md`
- `reports/exp44_badnoise_v4_states.csv`
- `reports/exp44_badnoise_v4_summary.json`
- `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_badnoise_v4_states.jsonl`

Next status target: `MINIMAX_STAGE2_HANDOFF_READY` or
`MINIMAX_STAGE2_HANDOFF_BLOCKED`.

## 2026-06-29 Stage2-Style Dataset Handoff

Status: `MINIMAX_STAGE2_DATA_HANDOFF_PARTIAL`.

Milestone F built the requested three Stage2-style dataset views from the
same-source pair and bad-noise v4 metadata:

- GT distillation: train/search/shadow = `24` / `8` / `8`;
- pseudo-success distillation: train/search/shadow = `24` / `8` / `8`;
- same-source preference: train/search/shadow = `24` / `8` / `8`;
- bad-noise states matched to pairs: `40` / `40`;
- scene overlap across train/search/shadow: `0`;
- first H20 experiment: pseudo-success SFT `30`-step;
- explicitly do not start with GT-only SFT.

The handoff is partial and marked `TRAINING_NOT_UNLOCKED` because the split
counts are below the requested minimum `train32/search16/shadow16`. This is a
debug/preflight handoff, not formal training-ready evidence.

Path validation caveat: the current Codex session does not have `/mnt/nas`
mounted, so absolute NAS paths inside the manifests could not be revalidated in
this session. H20 must verify every manifest path before running any dataloader
or runner.

No training, optimizer step, VOR-Eval use, hard comp, H20 modification, or
MiniMax quality-positive claim occurred.

Reports/manifests:

- `reports/exp44_stage2_dataset_handoff.md`
- `reports/exp44_stage2_dataset_handoff.csv`
- `reports/exp44_stage2_dataset_handoff_summary.json`
- `reports/exp44_h20_handoff_instructions.md`
- `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_stage2_gt_distill_train.jsonl`
- `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_stage2_pseudo_success_train.jsonl`
- `exp44_pai_minimax_targeted_same_source_mining/manifests/exp44_stage2_same_source_preference_train.jsonl`

Next status target: optional dataloader / one-batch forward smoke only if the
partial handoff is accepted and paths are verified on a mounted NAS environment.
