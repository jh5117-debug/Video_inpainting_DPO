# Codex CLI Handoff: Video_inpainting_DPO

Date: 2026-06-06

This file is the current handoff from the IDE-extension chat to a fresh Codex CLI session. Read this first, then verify live state from logs before acting. Do not rely on stale memories from earlier turns.

## 0. Execution Boundaries

PAI:

- Codex must not SSH into PAI and must not directly execute commands on PAI.
- Codex may only give the user complete copy/paste shell command blocks for PAI.
- The user manually runs PAI commands and pastes output back.
- Never say "I executed this on PAI".

H20:

- Codex may SSH to H20 when needed.
- H20 commands must be careful and scoped: do not kill unrelated jobs, do not delete data, do not overwrite outputs.
- H20 is/was generating Exp7-fix small-mask data; recheck live status before touching anything.

General:

- Do not start new long training unless the user explicitly asks and required gates are satisfied.
- Do not re-run completed Stage1 work for Exp8a; it already has usable `last_weights` and `dpo_diagnostics.csv`.
- Do not use VBench for partial-mask video inpainting. Use `tools/run_inpainting_metric_eval.py` / `inference/metrics.py`.

## 1. Repos And Important Paths

HAL repo root:

```text
/home/hj/H20_Video_inpainting_DPO_hal
```

PAI repo root:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO
```

H20 repo root:

```text
/home/nvme01/H20_Video_inpainting_DPO
```

GitHub repo:

```text
git@github.com:jh5117-debug/Video_inpainting_DPO.git
https://github.com/jh5117-debug/Video_inpainting_DPO.git
```

Important HAL files/directories:

```text
CODEX_CLI_HANDOFF.md
PRD/00_current_status.md
PRD/10_target_domain_youtubevos_davis_plan.md
PRD/12_exp8_regionloss_and_exp9_nolose_plan.md
experiment_registry/
reports/
scripts/
tools/
```

Important PAI paths:

```text
PAI SFT-48000 DiffuEraser:
/mnt/workspace/hj/nas_hj/weights/diffuEraser/converted_weights_step48000

PAI D3 root:
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4

PAI D3 comp manifest:
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4/manifests/selected_primary_comp.repaired.pai_paths.jsonl

PAI DAVIS eval data:
/mnt/workspace/hj/nas_hj/data/external/davis_432_240
```

Important H20 paths:

```text
H20 SFT-48000 DiffuEraser:
/home/nvme01/H20_Video_inpainting_DPO/weights/diffuEraser/converted_weights_step48000

H20 Exp7-fix smallmask data root:
/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/exp07_fix_videodpo_smallmask15_20_prior_k4
```

## 2. Current Experiment Lineage / Interpretation

Correct experiment narrative:

- Exp1: DiffuEraser reproduction / SFT / metric setting.
- Exp2: official VideoDPO VC2 reproduction.
- Exp3: official VideoDPO framework with DiffuEraser replacing VC2.
- Exp4: fullmask generated loser quality gate; data quality failed; no official DPO training conclusion.
- Old Exp5: D2 comp + plain DPO, beta500/beta10 plain; collapsed.
- New Exp5: D2 comp + winner-gap regularized DPO; improved relative to Old Exp5, not final success.
- New Exp6: D2 no-comp + same winner-gap regularized DPO. There was no plain Exp6.
- Exp7: changed task to partial-mask inpainting; current old Exp7 looked poor and suspicious, likely due to large masks / missing ProPainter prior / domain issues.
- Exp7-fix: regenerate VideoDPO small-mask 15%-20% + ProPainter-prior data before more DPO.
- Exp8a: target-domain D3 comp + full-loss regularized DPO, Stage1 2000 -> DAVIS val -> Stage2 2000 -> DAVIS val. This is the current active PAI pipeline.
- Exp8b: future region-weighted loss ablation. Do not conflate with Exp8a.
- Exp9: D3 comp/no-comp target-domain gates; current best earlier candidate was Exp9 D3-comp ckpt500, but direct longer DPO tends to degrade.

DPO loss notation for regularized DPO:

```text
m_w     = policy winner MSE
m_l     = policy loser MSE
m_w_ref = reference winner MSE
m_l_ref = reference loser MSE
win_gap  = m_w - m_w_ref
lose_gap = m_l - m_l_ref

L_total = -logsigmoid(-0.5 * beta_dpo * (win_gap - lose_gap_weight * lose_gap))
          + sft_reg_weight * m_w
          + winner_abs_reg_weight * m_w
          + winner_gap_reg_weight * ReLU(win_gap - winner_gap_reg_margin)
```

For New Exp5 / New Exp6 / Exp7 / Exp8a / Exp9 regularized runs:

```text
beta_dpo = 10
lose_gap_weight = 0.25
sft_reg_weight = 0.0
winner_abs_reg_weight = 0.05
winner_gap_reg_weight = 1.0
winner_gap_reg_margin = 0.0

L = -logsigmoid(-0.5 * 10 * (win_gap - 0.25 * lose_gap)) + 0.05 * m_w + ReLU(win_gap)
```

Exp8a uses `LOSS_REGION_MODE=full`: `m_w/m_l` are full-video/full-latent MSE, not region-weighted. Region weighting is postponed to Exp8b.

## 3. Current True PAI State: Exp8a

The user manually launched PAI Exp8a full-loss continuation. Current status comes from user-pasted audit at CST 2026-06-06 05:57.

Exp8a purpose:

```text
Run target-domain DAVIS validation around D3 comp DPO with ordinary full-loss regularized DPO:
Stage1 2000 -> Stage1 DAVIS val using DPO-S1 + SFT-S2 hybrid -> Stage2 2000 -> Stage2 DAVIS val.
```

Important: Stage1 training is complete. Do not rerun Stage1.

Stage1 completed run:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage1/20260605_142442_exp08_d3_comp_fullloss_wingap_lose025_s1_2000_davis_pai
```

Stage1 outputs confirmed:

```text
checkpoint-2000 exists
last_weights exists
dpo_diagnostics.csv exists, about 103K
```

Stage1 training timeline:

```text
Reached 2000/2000 at 2026-06-05 17:11 CST
checkpoint-2000 saved
last_weights saved
```

The original full-loss script initially failed during DAVIS validation due to incomplete local SD1.5 weights. Fixes already applied manually on PAI:

1. Added missing feature extractor config:

```text
/mnt/nas/hj/weights/stable-diffusion-v1-5/feature_extractor/preprocessor_config.json
```

2. Patched `diffueraser/diffueraser.py` to disable safety checker in `StableDiffusionDiffuEraserPipeline.from_pretrained`:

```python
safety_checker=None,
feature_extractor=None,
requires_safety_checker=False,
```

3. Re-launched continuation with `DAVIS_VIDEO_LENGTH=24` because 16 frames was too short for DiffuEraser/ProPainter prior. Previous error was:

```text
ValueError: The effective video duration is too short. Please make sure that the number of frames of video, mask, and priori is at least greater than 22 frames.
```

Current active PAI continuation:

```text
PID file:
logs/pipelines/exp08_d3_comp_fullloss_continue_after_s1_fixsafety_len24_pai.pid

PID at last audit:
809695

Log:
logs/pipelines/exp08_d3_comp_fullloss_continue_after_s1_fixsafety_len24_pai_20260606_054617.log

Current status at last audit:
Running Stage1 DAVIS validation, specifically DiffuEraser-base inference.
```

The log shows progress through DAVIS videos and saved many baseline 4-in-1 comparison videos, e.g.:

```text
.../exp08a_fullloss_stage1_val_davis_20260605_142442_continue_fixsafety_len24_20260606_054617/inference/DiffuEraser-base/bear/comparison_4in1.mp4
.../DiffuEraser-base/blackswan/comparison_4in1.mp4
.../DiffuEraser-base/bmx-bumps/comparison_4in1.mp4
...
```

Current Stage1 val output dir:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval/exp08a_fullloss_stage1_val_davis_20260605_142442_continue_fixsafety_len24_20260606_054617
```

At last audit, not yet seen:

```text
DPO-S1_SFT-S2 inference start
metrics/summary.csv
Stage2 start
Total optimization steps = 2000 for Stage2
```

So: Exp8a is not finished. It is currently in Stage1 DAVIS validation. If it finishes Stage1 val, the continuation script should automatically start Stage2.

## 4. Current H20 State: Exp7-Fix Smallmask Data

H20 originally ran Exp7-fix small-mask data generation on GPUs 0-5, then the user asked to leave GPU0 free and use GPUs 1-7. We switched it safely:

- Old process group `502994` was terminated.
- Existing shard root was preserved.
- New resume process was launched with `GPUS=1,2,3,4,5,6,7` and the same shard root.
- Already completed shards are skipped via `.done`; this is not a full restart.
- GPU0 was confirmed free in the last check.

New H20 resume state from last successful SSH checks:

```text
PID: 3433414
Log: /home/nvme01/H20_Video_inpainting_DPO/logs/pipelines/exp07_fix_smallmask_prior_data_generation_h20_gpus1_7_20260605_133007.log
PID file: /home/nvme01/H20_Video_inpainting_DPO/logs/pipelines/exp07_fix_smallmask_prior_data_generation_h20_gpus1_7.pid
Latest log pointer: /home/nvme01/H20_Video_inpainting_DPO/logs/pipelines/exp07_fix_smallmask_prior_data_generation_h20_gpus1_7.latest_log
Report: /home/nvme01/H20_Video_inpainting_DPO/reports/h20_exp07_fix_smallmask_prior_gpu_switch_report.md
Shard root: /home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/exp07_fix_videodpo_smallmask15_20_prior_k4/_shards_gpu0_5_20260605_110124
```

Progress sample at the time:

```text
done=30
failed=0
dirs=37
```

This state must be rechecked live before any H20 action. Do not start H20 Stage1 until selection output exists.

Selection-gate condition:

Only consider H20 Exp7-fix Stage1 after one of these exists and has rows:

```text
/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/exp07_fix_videodpo_smallmask15_20_prior_k4/manifests/selected_primary_comp.jsonl
/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/exp07_fix_videodpo_smallmask15_20_prior_k4/manifests/selected_primary_comp.repaired.jsonl
```

## 5. What The New Codex CLI Should Do First

From HAL repo:

```bash
cd /home/hj/H20_Video_inpainting_DPO_hal
pwd
git branch --show-current
git status --short
sed -n '1,260p' CODEX_CLI_HANDOFF.md
sed -n '1,220p' PRD/00_current_status.md
sed -n '1,220p' PRD/12_exp8_regionloss_and_exp9_nolose_plan.md
```

For PAI, the new Codex cannot run directly. Ask the user to run a PAI audit command if live state is needed. A safe PAI audit for Exp8a is below.

For H20, the new Codex may SSH and run read-only checks if needed.

## 6. Safe PAI Audit Command For Exp8a

Give this to the user when checking PAI Exp8a. It is read-only:

```bash
cd /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO
set +e

PLOG=$(ls -t logs/pipelines/exp08_d3_comp_fullloss_continue_after_s1_fixsafety_len24_pai_*.log 2>/dev/null | head -1)
PID_FILE=logs/pipelines/exp08_d3_comp_fullloss_continue_after_s1_fixsafety_len24_pai.pid
VAL1=$(find /mnt/nas/hj/H20_Video_inpainting_DPO/logs/target_eval \
  -maxdepth 1 -type d -name '*exp08a_fullloss_stage1_val_davis*fixsafety_len24*' \
  -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

echo DATE=$(date)
echo PLOG=$PLOG
echo PID_FILE=$PID_FILE
[ -f "$PID_FILE" ] && echo PID=$(cat "$PID_FILE")
[ -f "$PID_FILE" ] && ps -fp "$(cat "$PID_FILE")" || true

echo GPU
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

echo KEY_LOG
grep -a -nE 'Exp8|Exp8a|Stage1|Stage2|DAVIS|DPO-S1_SFT-S2|DiffuEraser-base|metrics|summary.csv|side_by_side|Total optimization steps|Epoch|global_step=|checkpoint|last_weights|Traceback|FAILED|ERROR|OutOfMemory|SIGFPE|ValueError|RuntimeError|OSError|No such file' "$PLOG" | tail -220 || true

echo VAL1=$VAL1
if [ -n "$VAL1" ]; then
  echo sample_mp4_count=$(find "$VAL1/inference" -name '*.mp4' 2>/dev/null | wc -l)
  echo side_by_side_count=$(find "$VAL1/side_by_side" -name '*.mp4' 2>/dev/null | wc -l)
  ls -lh "$VAL1/metrics/summary.csv" "$VAL1/pair_manifest.csv" "$VAL1/index.html" "$VAL1/report.md" 2>/dev/null || true
fi

echo STAGE2_DIRS
find /mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/stage2 \
  -maxdepth 1 -type d -name '*exp08_d3_comp_fullloss*' \
  -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort | tail -5

echo AUDIT_DONE
```

Interpretation:

- PID alive + GPU0 memory/utilization: still running DAVIS inference/validation.
- Log shows `DPO-S1_SFT-S2`: baseline finished; current model validation started.
- Log shows `metrics/summary.csv`: Stage1 validation metrics generated.
- Log shows `Stage2 start` / `Total optimization steps = 2000`: Stage2 training started.
- Stage2 `last_weights` + Stage2 validation `summary.csv`: full pipeline finished.
- Any `Traceback` / `ERROR` / `OSError` / `ValueError`: stop and diagnose; do not restart Stage1.

## 7. Safe H20 Read-Only Check

Use SSH only if necessary:

```bash
ssh -i ~/.ssh/codex_h20_2 -o BatchMode=yes -o ConnectTimeout=15 ubuntu@27.190.15.128 '
cd /home/nvme01/H20_Video_inpainting_DPO || exit 2
set +e
DATA=data/generated_losers/exp07_fix_videodpo_smallmask15_20_prior_k4
PID=$(cat logs/pipelines/exp07_fix_smallmask_prior_data_generation_h20_gpus1_7.pid 2>/dev/null)
LOG=$(cat logs/pipelines/exp07_fix_smallmask_prior_data_generation_h20_gpus1_7.latest_log 2>/dev/null)
echo DATE=$(date)
echo PID=$PID
echo LOG=$LOG
ps -fp "$PID" 2>/dev/null || echo generation_pid_not_running
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
ls -lh "$DATA"/manifests 2>/dev/null || true
for f in "$DATA"/manifests/*.jsonl; do [ -f "$f" ] && echo "$(basename "$f") $(wc -l < "$f")"; done
for f in "$DATA"/manifests/selected_primary_comp.repaired.jsonl "$DATA"/manifests/selected_primary_comp.jsonl; do [ -f "$f" ] && wc -l "$f" || echo MISSING "$f"; done
SHARDS="$DATA/_shards_gpu0_5_20260605_110124"
echo done=$(find "$SHARDS" -name .done 2>/dev/null | wc -l) failed=$(find "$SHARDS" -name .failed 2>/dev/null | wc -l) dirs=$(find "$SHARDS" -maxdepth 1 -type d -name "shard_*" 2>/dev/null | wc -l)
tail -n 80 "$LOG" 2>/dev/null || true
'
```

## 8. Explicit Do-Not-Do List

Do not:

- Do not start PAI commands directly.
- Do not rerun Exp8a Stage1.
- Do not delete old PAI target_eval outputs; failed validation attempts are useful evidence.
- Do not start H20 Stage1 before selected manifests exist.
- Do not delete H20 raw / comp / ProPainter / shard outputs.
- Do not regenerate all Exp7-fix data from scratch unless explicitly approved.
- Do not train DPO Stage2 except as part of the already-launched PAI Exp8a continuation if it reaches that step.
- Do not use VBench for inpainting.
- Do not use ordinary base DiffuEraser for YouTube-VOS/D3/DAVIS; use SFT-48000.

## 9. Suggested First Prompt For A New Codex CLI

Copy this to the new Codex CLI after starting it in `/home/hj/H20_Video_inpainting_DPO_hal`:

```text
请先阅读 CODEX_CLI_HANDOFF.md，并严格按其中执行边界接手。当前不要启动新训练，不要删除或重生成数据，不要重跑 Exp8a Stage1。先总结你读到的当前状态：PAI Exp8a full-loss regularized DPO Stage1 已完成，正在/刚刚在做 len24 DAVIS Stage1 validation；H20 Exp7-fix smallmask 数据生成已切到 GPU1-7 但需要只读复查。PAI 你不能直接执行，只能给我可复制命令；H20 你可以 SSH，但先只读检查。第一步请给我：1) 你读取 handoff 后的状态摘要；2) PAI Exp8a 的下一条安全审查命令；3) H20 smallmask 生成的只读检查命令或你自行 SSH 检查结果。不要启动训练。
```
