# Exp27 Candidate Review Round 2

## Reviewer A

Nearest-work overlap remains highest with LocalDPO. RC-FPO survives only because it is task-native and inpainting-specific: real task masks, OR affected regions, and outside-context anti-shortcut checks are not LocalDPO's random corruption setup.

## Reviewer B

RC-FPO's strict claims must be empirical and algorithmic, not theoretical winner-preservation. Region weighting can be mathematically described as gradient reallocation, but no monotone winner theorem follows. SDPO-based variants remain heuristic unless a new theorem is added.

## Reviewer C

RC-FPO is implementable from Exp25/26 without changing shared trainers. SDPO and Linear-DPO exact baselines need more save/resume and gradient parity work before longer runs. LocalDPO-style data baseline must be implemented faithfully.

## Reviewer D

RC-FPO has a clean experimental ladder: data-only, region-only, data+region, then objective baselines. VOR-Eval and DAVIS50/YouTubeVOS100 must remain final-only. Search-dev/shadow-dev splits are mandatory.

## Reviewer E

Reject text if paper claims "LocalDPO for inpainting plus SDPO plus Linear-DPO." Accept condition: beat faithful LocalDPO-adapted inpainting baseline with equal data/compute and show improvements inside mask/boundary/affected regions without outside shortcuts.

## Decision

Primary: `RC-FPO`, Restoration-Critical Failure-Structured Preference Optimization.

Fallback: `ST-Pref`, Stage-Aware Spatial/Temporal Preference Decomposition.

Paused as baselines: Region-SDPO, Linear-Frozen, Linear-EMA.
