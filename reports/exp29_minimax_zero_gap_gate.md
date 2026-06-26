# Exp29 MiniMax Zero-Gap Gate

Date: 2026-06-26

Status: `MINIMAX_ZERO_GAP_PASSED`

Policy and reference were initialized from the same MiniMax checkpoint and
evaluated on the same flow-matching preference row.

| Field | Value |
| --- | ---: |
| winner policy loss | 0.0171425510 |
| winner reference loss | 0.0171425510 |
| loser policy loss | 0.0140384063 |
| loser reference loss | 0.0140384063 |
| win gap | 0.0 |
| lose gap | 0.0 |
| preference margin | 0.0 |
| DPO loss | 0.6931471825 |

The loss is numerically `log(2)` and both gaps are zero, so the MiniMax DPO
plumbing satisfies the zero-gap identity check.

