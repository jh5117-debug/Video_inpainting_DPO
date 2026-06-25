# PAI Post-Maintenance Permission Blockers

Detected after PAI maintenance on 2026-06-25.

Blockers requiring PAI root terminal:

- `hj` cannot read/traverse `/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000`.
  - Blocks Exp25 DiffuEraser root-cause matrix.
  - Required fix: ACL read/traverse only on this concrete asset subtree.
- `hj` cannot write Exp-only experiment output directories:
  - `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp25_vor_or_preference_data`
  - `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp26_videopainter_dpo_v2`
  - `/mnt/nas/hj/H20_Video_inpainting_DPO/experiments/dpo/exp27_paper_grounded_preference_study`
  - Required fix: create/own only these experiment-specific directories, not shared roots.

Unsafe operations intentionally avoided:

- no `chmod 777`
- no recursive ownership change of `/mnt/nas/hj`
- no recursive ownership change of `/mnt/nas/hj/weights`
- no modification of shared trainer or `inference/metrics.py`
