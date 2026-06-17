# Exp16 Preflight Report

status: `blocked`
blocked_reason: `manifest lacks real ProPainter prior paths; build exp16 prior cache first`

## Manifest Audit

- manifest: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/official_videodpo_diffueraser_youtubevos_partialmask_loser_k4/manifests/selected_primary_comp.repaired.pai_paths.jsonl`
- exists: False
- total audited rows: 0
- rows with prior: 0
- missing prior: 0

## Tensor Sanity

- tensor sanity: `passed`
- extra loss: 0.002685723826289177
- prior_conf_mean: 0.22777195274829865
- L_prior: 0.02560148574411869
- L_gen: 0.0008679062593728304
- L_boundary_extra: 0.0008217995055019855

This preflight does not launch training. Exp16 training remains blocked
until every row has a real ProPainter prior path and the full trainer
integrates `z_hat_x0`, `z_prior`, and `z_gt` into total loss.
