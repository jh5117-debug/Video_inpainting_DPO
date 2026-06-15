# Status

Status: blocked_before_preflight.

PAI sync strategy: clean_worktree.

Clean repo:
`/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp14_videopainter_gate`

What passed:
- Exp14 trainer and launcher are present.
- Static checks pass.
- VideoPainter code repo is present.
- Data and manifest are present.
- GPUs are available.

Blocker:
- Missing `CogVideoX-5b-I2V` base model.
- Missing VideoPainter branch checkpoint.

No preflight, gate2000 training, dpo_diag, checkpoint, or DAVIS eval was run.
