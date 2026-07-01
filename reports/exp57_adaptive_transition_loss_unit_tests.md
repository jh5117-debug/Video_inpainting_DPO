# Exp57 Adaptive Transition Loss Unit Tests

Status: `EXP57_ADAPTIVE_TRANSITION_LOSS_READY`

Unit tests cover:

- loser lambda becomes zero when loser gradient conflicts with winner;
- loser lambda remains positive and clipped when aligned;
- transition risk increases preservation and downscales object DPO;
- backtracking reduces update scale when overlap worsens;
- backtracking rejects unsafe updates;
- zero-gap DPO loss is near `log(2)`;
- outside has no loser DPO by default.
