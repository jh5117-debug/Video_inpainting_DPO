# Exp60B VPData Subset Transfer To PAI/NAS

Status: `EXP60B_TRANSFER_BLOCKED`

Transfer to PAI/NAS was not run because no source lane produced the exact
locked train1000/test100 subset.

## Gate State

- H20 hf-mirror route: 1,089 / 1,100 raw videos downloaded.
- H20 clash proxy fallback: 1,089 / 1,100 raw videos downloaded.
- HAL fallback: 11 missing URLs probed, 0 reachable, 11 HTTP 403.
- Required ready condition: 1,100 / 1,100 raw videos with sha256.
- Current ready condition: not met.

## Decision

No PAI manifest was generated and no PAI `EXP60B_PAI_VPDATA_SUBSET_READY`
status was written. The partial H20 staging set can be preserved for audit, but
it is not sufficient to start D3 mask generation, loser generation, inference,
or DPO.

No training, inference, loser generation, GPU job, full VPData download, or data
replacement was run.
