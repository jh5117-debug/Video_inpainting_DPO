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
