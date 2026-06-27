# Exp31 VideoPainter Checkpoint Ladder Support

Status: `VIDEOPAINTER_2000_CHECKPOINT_LADDER_READY`

The isolated Exp31 trainer now supports an explicit checkpoint ladder with:

`--checkpoint_steps 0,1,10,50,100,200,500,1000,1500,2000`

This preserves the requested audit checkpoints without saving every step.
Explicit steps are protected from retention pruning. Periodic checkpoints can be
disabled with `--checkpointing_steps 0` when an explicit list is supplied.

Validation:

- `git diff --check`: passed
- `python -m py_compile exp26_videopainter_dpo_v2/code/*.py exp26_videopainter_dpo_v2/tests/*.py`: passed
- `python -m unittest discover -s exp26_videopainter_dpo_v2/tests -p 'test_*.py'`: 28 tests passed
- `bash -n`: passed

No Exp26, Exp30, shared trainer, or `inference/metrics.py` files were modified.
