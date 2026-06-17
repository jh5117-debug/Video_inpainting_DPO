# Exp18 Multi-frame Propagation-Confidence Gated DPO

Exp18 continues from the current best `Exp11 outer b0.75 S2`, but replaces the
old Exp16 GT-error prior confidence with real multi-frame propagation
confidence.

Core idea:

```text
propagatable mask pixels -> preserve propagated pixels
non-propagatable mask pixels -> generate with GT/context preference
outer boundary -> keep Exp11 boundary-aware seam constraint
```

Current status:

```text
IMPLEMENTATION_READY_ON_HAL
PAI_RUN_BLOCKED_IN_THIS_SESSION_BY_MISSING_PAI_MOUNT_OR_SSH
```

Run order on PAI:

```bash
bash exp18_multiframe_propagation_gated_dpo/scripts/launch_exp18_overnight_pai.sh
```

