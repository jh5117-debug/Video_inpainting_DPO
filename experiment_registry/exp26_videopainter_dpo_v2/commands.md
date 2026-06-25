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
exp26_videopainter_dpo_v2/manifests/vp2_gate64_primary32_visual_reviewed_comp.jsonl
```

Do not launch DPO micro-training until `hj` can write:

```text
/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp26_videopainter_dpo_v2
```
