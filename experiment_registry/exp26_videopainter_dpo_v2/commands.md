# Exp26 Commands

Static validation:

```bash
python -m py_compile exp26_videopainter_dpo_v2/code/*.py
python -m unittest discover -s exp26_videopainter_dpo_v2/tests -p 'test_*.py'
bash -n exp26_videopainter_dpo_v2/scripts/*.sh
```

Post-maintenance Gate64 repair artifacts:

```bash
python exp26_videopainter_dpo_v2/code/audit_gate64_duplicate_sources_deep.py
```

Primary manifest:

```text
exp26_videopainter_dpo_v2/manifests/vp2_gate64_primary32_final.jsonl
```

Gate64 final temporal review evidence:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/gate64_final_temporal_review_20260625
```

Next allowed milestone:

```text
VP-L0/L1 one-batch and one-step DPO adapter validation on the final primary-32 manifest.
```

Primary-32 10-step gate:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/run_step1_step10_eval_20260625.sh
```

Outputs:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp26_videopainter_dpo_v2/vp_primary32_10step_retry1_20260625_145257
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/searchdev_step1_vp_primary32_10step_retry1_20260625_151020
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/searchdev_step10_vp_primary32_10step_retry1_20260625_151020
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/reports_step1_step10_20260625_151020
```

Primary-32 50-step gate:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/run_vp_primary32_50step_gate_20260625.sh
```

Outputs:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp26_videopainter_dpo_v2/vp_primary32_50step_20260625_171032
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/searchdev_step10_vp_primary32_50step_20260625_171032
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/searchdev_step30_vp_primary32_50step_20260625_171032
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/searchdev_step50_vp_primary32_50step_20260625_171032
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/reports_vp_primary32_50step_20260625_171032
```
## 2026-06-26 Shadow-Dev Confirmatory Validation

Main controller was run on PAI from the right-side Exp26 runtime snapshot:

```bash
RUN_ROOT=/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/shadowdev_confirmatory_20260625 \
PYTHON_BIN=/usr/local/bin/python \
bash exp26_videopainter_dpo_v2/scripts/run_vp2_shadowdev_confirmatory_pai.sh
```

Primary-only TC/VFID diagnostics:

```bash
CUDA_VISIBLE_DEVICES=5 python exp26_videopainter_dpo_v2/code/shadowdev_confirmatory_analysis.py \
  --run-root /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/shadowdev_confirmatory_20260625 \
  --device cuda \
  --tc-vfid-ranges no_first_frame \
  --tc-vfid-steps step0,step50 \
  --tc-vfid-variants raw,comp \
  tc-vfid
```

Seed robustness reused the fixed 16-row manifest and changed only the
VideoPainter inference seed for Step0/Step50:

```bash
python exp26_videopainter_dpo_v2/code/run_vp2_gate64_official_generation.py \
  --manifest seed_robustness_16_manifest.jsonl \
  --limit 16 \
  --num-frames 49 \
  --num-inference-steps 20 \
  --guidance-scale 6.0 \
  --seed 20260620
```

Post-confirmation sanity/readback:

```bash
git fetch --all --prune
git branch --show-current
git rev-parse HEAD
git status --short
git diff --check
```

Reports:

```text
reports/exp26_postconfirmation_readback.md
reports/exp26_postconfirmation_sanity_audit.md
reports/exp26_postconfirmation_sanity_audit.csv
reports/exp26_postconfirmation_sanity_audit.json
```

External 49F source inventory:

```bash
python exp26_videopainter_dpo_v2/code/postconfirmation_external_inventory.py \
  --project-root "$PWD" \
  --target-rows 32
```

Artifacts:

```text
reports/exp26_external_49f_inventory.md
reports/exp26_external_49f_inventory.csv
reports/exp26_external_49f_inventory.json
exp26_videopainter_dpo_v2/manifests/vp2_external_49f_candidate_inventory.jsonl
exp26_videopainter_dpo_v2/manifests/vp2_external_49f_validation_16_or_32.jsonl
```

External validation preregistration was executed on PAI before any external
model output generation:

```bash
python /tmp/exp26_external_prereg_code/postconfirmation_external_preregister.py \
  --source-manifest /tmp/exp26_external_prereg_code/vp2_external_49f_validation_16_or_32.jsonl \
  --run-root /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/postconfirmation_20260626/external_validation/preregistered \
  --output-manifest /tmp/exp26_external_prereg_out/manifests/vp2_external_validation_preregistered.jsonl \
  --mask-manifest /tmp/exp26_external_prereg_out/manifests/vp2_external_validation_masks.jsonl \
  --status-csv /tmp/exp26_external_prereg_out/reports/exp26_external_validation_preregistration_status.csv \
  --report-md /tmp/exp26_external_prereg_out/reports/exp26_external_validation_preregistration.md \
  --report-json /tmp/exp26_external_prereg_out/reports/exp26_external_validation_preregistration.json
```

Artifacts:

```text
reports/exp26_external_validation_preregistration.md
reports/exp26_external_validation_preregistration.json
reports/exp26_external_validation_preregistration_status.csv
exp26_videopainter_dpo_v2/manifests/vp2_external_validation_preregistered.jsonl
exp26_videopainter_dpo_v2/manifests/vp2_external_validation_masks.jsonl
/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/postconfirmation_20260626/external_validation/preregistered
```

External validation generation used task-level checkpoint parallelism only.
Each checkpoint used a single GPU and the same preregistered manifest:

```bash
bash exp26_videopainter_dpo_v2/scripts/run_vp2_external_validation_checkpoint_pai.sh \
  step0 0

bash exp26_videopainter_dpo_v2/scripts/run_vp2_external_validation_checkpoint_pai.sh \
  step50 5

bash exp26_videopainter_dpo_v2/scripts/run_vp2_external_validation_checkpoint_pai.sh \
  step30 6

bash exp26_videopainter_dpo_v2/scripts/run_vp2_external_validation_checkpoint_pai.sh \
  step10 7
```

Generation/leakage audit:

```bash
python exp26_videopainter_dpo_v2/code/postconfirmation_external_analysis.py status \
  --run-root /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/postconfirmation_20260626/external_validation \
  --manifest /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/postconfirmation_20260626/external_validation/preregistered/manifests/vp2_external_validation_preregistered.jsonl \
  --report-dir /tmp/exp26_external_analysis_out/reports \
  --project-root /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp26_videopainter

python exp26_videopainter_dpo_v2/code/postconfirmation_external_analysis.py leakage \
  --run-root /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/postconfirmation_20260626/external_validation \
  --manifest /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/postconfirmation_20260626/external_validation/preregistered/manifests/vp2_external_validation_preregistered.jsonl \
  --report-dir /tmp/exp26_external_analysis_out/reports \
  --project-root /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp26_videopainter
```

External validation metrics/statistics:

```bash
python exp26_videopainter_dpo_v2/code/postconfirmation_external_analysis.py metrics \
  --run-root /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/postconfirmation_20260626/external_validation \
  --manifest /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/postconfirmation_20260626/external_validation/preregistered/manifests/vp2_external_validation_preregistered.jsonl \
  --report-dir /tmp/exp26_external_analysis_out/reports \
  --project-root /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp26_videopainter \
  --device cuda

python exp26_videopainter_dpo_v2/code/postconfirmation_external_analysis.py stats \
  --run-root /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/postconfirmation_20260626/external_validation \
  --manifest /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/postconfirmation_20260626/external_validation/preregistered/manifests/vp2_external_validation_preregistered.jsonl \
  --report-dir /tmp/exp26_external_analysis_out/reports \
  --project-root /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp26_videopainter \
  --device cuda

python exp26_videopainter_dpo_v2/code/postconfirmation_external_analysis.py tc-vfid \
  --run-root /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/postconfirmation_20260626/external_validation \
  --manifest /mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp26_videopainter_dpo_v2/postconfirmation_20260626/external_validation/preregistered/manifests/vp2_external_validation_preregistered.jsonl \
  --report-dir /tmp/exp26_external_analysis_out/reports \
  --project-root /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp26_videopainter \
  --device cuda
```
