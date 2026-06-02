# Target-Domain YouTube-VOS / DAVIS Plan

## Domain Boundary

Final target domain:

- YouTube-VOS
- DAVIS

VideoDPO is now treated as a bridge domain only. Its role is to validate native
VideoDPO repository integration, DiffuEraser loading, generated-loser manifests,
partial-mask plumbing, winner-gap regularization, DPO diagnostics, and
Stage1/Stage2 weight loading. VideoDPO partial-mask evaluation is diagnostic
only and must not be reported as final quality.

Do not do a VideoDPO partial-mask SFT warmup. If SFT is needed, it should be on
target-domain data or closely related inpainting data.

## Current Interpretation

- Exp3 validates replacing VC2 with DiffuEraser in the native VideoDPO repo.
- Exp5 and Exp6 validate generated-loser data-only DPO and stabilization
  tricks. New Exp5 winner-anchor improves optimization stability but is not a
  final visual success.
- Exp7 validates partial-mask task support inside the native
  VideoDPO-DiffuEraser pipeline. Its VideoDPO partial-mask eval is a diagnostic,
  not final success.
- The current priority is target-domain evaluation of existing checkpoints on
  YouTube-VOS / DAVIS.

## D3 Target-Domain Data

H20 D3 root:

```text
/home/nvme01/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4
```

PAI target root:

```text
/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4
```

H20 audit from 2026-06-02:

- size: 249G
- files: 1,819,879
- shards: 3,327
- `selected_primary_comp.jsonl`: 3,327 rows
- `selected_primary_nocomp.jsonl`: 3,327 rows
- sampled 100 selected-primary-comp rows: all status `OK`
- sampled win/mask/final loser paths: 16 frames, 512x320, readable
- all sampled manifest paths are H20-only `/home/nvme01/...` absolute paths

Therefore D3 must be path-rewritten on PAI before training. Do not train
directly from the original H20-path manifests.

## D3 Sync Strategy

Use PAI-side pull from H20:

```bash
bash scripts/sync_d3_from_h20_to_pai.sh
```

Default `SYNC_MODE=slim` syncs only selected-primary manifests and the selected
primary paths required for first Exp9 training. If slim packaging proves risky,
run:

```bash
SYNC_MODE=full bash scripts/sync_d3_from_h20_to_pai.sh
```

The sync script uses `--partial --append-verify`, never uses `--delete`, and
writes logs under `logs/data_sync/` when launched through `nohup`.

After sync:

```bash
python tools/d3_post_generation_audit_and_repair.py
python tools/d3_training_readiness_check.py
```

Expected reports:

```text
<D3_ROOT>/reports/d3_post_generation_audit.md
<D3_ROOT>/reports/d3_training_readiness_report.md
```

Expected repaired manifests:

```text
manifests/selected_primary_comp.repaired.jsonl
manifests/selected_primary_nocomp.repaired.jsonl
manifests/selected_primary_comp.repaired.pai_paths.jsonl
manifests/selected_primary_nocomp.repaired.pai_paths.jsonl
```

Readiness split:

- D3 full readiness can remain false when the PAI slim sync omits secondary
  manifests.
- D3 primary-comp gate readiness is the relevant condition for the first Exp9
  Stage1 gate.
- Use `tools/d3_primary_comp_gate_readiness_check.py` to confirm
  `ready_primary_comp_gate=true`; do not let secondary-manifest absence block
  the selected-primary-comp gate.

## Target-Domain Eval Gate

Run preflight first:

```bash
bash scripts/eval_target_youtubevos_davis_checkpoints_metricpy.sh
```

The target eval must use the previous DiffuEraser best settings:

- denoise steps = 6
- no PCM
- no Gaussian blur
- no unnecessary mask dilation
- frame-wise output / metric path
- hard comp outside mask

If the current eval backend cannot guarantee these settings, write a report and
do not pretend the eval is valid.

Metric backend:

- Use VBench only for video generation / full-mask prompt generation.
- Use the project metric module for YouTube-VOS / DAVIS partial-mask video
  inpainting.
- No exact `metric.py` exists in this checkout; the existing metric module is
  `inference/metrics.py`.
- `tools/run_inpainting_metric_eval.py` is the thin adapter that imports the
  metric module and writes `summary.csv`, `summary.json`, and `summary.md`.
- Do not reimplement PSNR, SSIM, LPIPS, or temporal metrics in target eval
  scripts.

Compare existing checkpoints:

- DiffuEraser-base / current best SFT DiffuEraser
- Exp3 official_videodpo_diffueraser checkpoint
- new Exp5 winner-anchored DPO checkpoint
- new Exp6 winner-anchored DPO checkpoint if available
- Exp7 DPO-S1 + DPO-S2
- DPO-S1 + SFT-S2 hybrid

Report:

```text
reports/target_domain_youtubevos_davis_eval_report.md
```

The report must answer whether any VideoDPO-bridge DPO checkpoint transfers to
YouTube-VOS / DAVIS and whether DPO-S1 + SFT-S2 beats DPO-S1 + DPO-S2.

## Exp9 Plan Boundary

Do not start Exp9 automatically. Exp9 starts only if D3 selected-primary
readiness is true and target-domain evaluation shows that VideoDPO-bridge DPO
does not transfer or target-domain DPO data is required.

Exp9 first gate:

- data: D3 `selected_primary_comp.repaired.pai_paths.jsonl` if path rewrites
  were needed, otherwise `selected_primary_comp.repaired.jsonl`
- task: partial-mask training
- `train_mask_mode=partial`
- `mask_from_manifest=true`
- `loss_region_mode=full` first
- winner-anchored DPO
- beta = 10
- `winner_abs_reg_weight=0.05`
- `winner_gap_reg_weight=1.0`
- `lose_gap_weight=0.25` or no-lose gate
- Stage1 DPO only
- frozen SFT / target-domain Stage2
- no DPO Stage2
- 1000 or 1500 step gate before any long run

Prepared launchers:

```text
scripts/launch_exp9_youtubevos_d3_partialmask_wingap_stage1_gate_pai.sh
scripts/launch_exp9_youtubevos_d3_partialmask_wingap_nolose_stage1_gate_pai.sh
```

The no-lose script is a fallback only and must not be launched unless explicitly
requested.

## Do Not Do

- Do not start Exp9 training now.
- Do not jump to Exp8.
- Do not continue VideoDPO warmup.
- Do not run full VBench.
- Do not regenerate D2.
- Do not delete or overwrite D3 H20 originals.
- Do not train from manifests containing `/home/nvme01/...` paths on PAI.
- Do not treat VideoDPO partial-mask eval as final result.
