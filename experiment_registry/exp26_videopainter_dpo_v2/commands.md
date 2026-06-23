# Exp26 Commands

Static validation:

```bash
python -m py_compile exp26_videopainter_dpo_v2/code/*.py
python -m unittest discover -s exp26_videopainter_dpo_v2/tests -p 'test_*.py'
bash -n exp26_videopainter_dpo_v2/scripts/*.sh
```

No GPU launcher is enabled yet. The next step is L0/L1 official VideoPainter
baseline and native loss parity, not a 2000-step DPO run.
