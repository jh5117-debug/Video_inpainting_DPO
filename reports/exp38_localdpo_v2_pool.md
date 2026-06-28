# Exp37 LocalDPO-style OR Corruption Pool

Status: `PENDING_CODEX_VISUAL_REVIEW`

Built deterministic local corruptions from VOR-Train style rows only:

- condition = `V_obj`
- winner = `V_bg`
- loser = locally corrupted `V_bg`
- mask = object mask
- affected map = `abs(V_obj - V_bg)` for profile construction
- candidate rows per source <= 2
- VOR-Eval used = `false`

Train selected summary: `{'selected_rows': 32, 'candidate_rows': 64, 'classification_counts': {'HARD_BUT_PLAUSIBLE': 3, 'MEDIUM_HARD_ELIGIBLE': 27, 'TRIVIAL_BAD': 2}, 'profile_counts': {'V2_object_mild': 13, 'V2_object_micro': 8, 'V2_object_affected_local': 6, 'V2_boundary_effect_mild': 5}, 'usable_selected': 30, 'outside_mae_mean': 0.06248566812909376, 'scene_groups': 32}`
Heldout selected summary: `{'selected_rows': 16, 'candidate_rows': 32, 'classification_counts': {'MEDIUM_HARD_ELIGIBLE': 12, 'TRIVIAL_BAD': 3, 'HARD_BUT_PLAUSIBLE': 1}, 'profile_counts': {'V2_object_mild': 5, 'V2_object_micro': 4, 'V2_boundary_effect_mild': 4, 'V2_object_affected_local': 3}, 'usable_selected': 13, 'outside_mae_mean': 0.24334227829231253, 'scene_groups': 16}`

Codex visual review is required before marking the pool ready.
