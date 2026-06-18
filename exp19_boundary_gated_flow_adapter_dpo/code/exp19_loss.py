#!/usr/bin/env python3
"""Exp19 loss contract.

Exp19 intentionally keeps the Exp11 outer b0.75 S2 DPO loss unchanged. This
module only documents that contract so launchers do not silently introduce an
extra flow target loss.
"""

EXP19_LOSS_CONTRACT = {
    "base_loss": "Exp11 region-local normalized-gap clipped-loser-gap winner-anchored DPO",
    "flow_loss_exp19a": None,
    "flow_loss_exp19b": None,
    "flow_loss_exp19c": "optional latent warp consistency only if z_hat0 path is safe",
}
