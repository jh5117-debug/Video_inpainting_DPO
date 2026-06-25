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
