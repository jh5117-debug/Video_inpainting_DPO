# Exp20 Legacy Full Parity

- manifest: `/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO/data/generated_losers/exp09_10_11_youtubevos_gtwin_d3comp_pai/manifests/selected_primary_comp.gtwin.pai_paths.jsonl`
- sample_id: `youtubevos_000000_003234408d`
- mask_id: `mask_000`
- status: `LEGACY_FULL_PARITY_PASSED`

## Summary

- legacy map max_abs_diff: `0`
- prediction-grad cosine: `1.00011944771`
- prediction-grad relative L2: `0`

Model-parameter backward and one-step optimizer delta parity are deferred to the real 10-step smoke because this lightweight harness does not instantiate two full training models simultaneously.

## Scalar Parity

| metric | old | new | abs_diff |
| --- | ---: | ---: | ---: |
| model_losses_w | 2.0019614696502686 | 2.0019614696502686 | 0.0 |
| model_losses_l | 2.0128071308135986 | 2.0128071308135986 | 0.0 |
| ref_losses_w | 2.0093865394592285 | 2.0093865394592285 | 0.0 |
| ref_losses_l | 2.022118330001831 | 2.022118330001831 | 0.0 |
| raw_win_gap | -0.007424980401992798 | -0.007424980401992798 | 0.0 |
| raw_lose_gap | -0.009311072528362274 | -0.009311072528362274 | 0.0 |
| norm_win_gap | -0.0038020717911422253 | -0.0038020717911422253 | 0.0 |
| norm_lose_gap | -0.004477417562156916 | -0.004477417562156916 | 0.0 |
| norm_lose_gap_clipped | -0.004477417562156916 | -0.004477417562156916 | 0.0 |
| inside_term | 0.013413587585091591 | 0.013413587585091591 | 0.0 |
| dpo_loss | 0.6888894438743591 | 0.6888894438743591 | 0.0 |
| winner_abs_reg | 2.0019614696502686 | 2.0019614696502686 | 0.0 |
| winner_gap_reg | 0.009930596686899662 | 0.009930596686899662 | 0.0 |
| total_loss | 0.7989181280136108 | 0.7989181280136108 | 0.0 |
| implicit_acc | 1.0 | 1.0 | 0.0 |
| loser_degrade_ratio | 0.0 | 0.0 | 0.0 |
| weight_map_max_abs_diff |  |  | 0.0 |
| prediction_grad_cosine |  |  | -0.00011944770812988281 |
| prediction_grad_relative_l2 |  |  | 0.0 |
