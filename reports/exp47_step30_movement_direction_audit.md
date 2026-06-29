# Exp47 Step30 Movement Direction Audit

Status: `EXP47_STEP30_MOVEMENT_AUDITED`
Root signal: `SFT_LOSS_OR_TARGET_PATH_BUG`

Rows audited: `48` search/shadow rows. This audit is read-only and uses sampled frame-space metrics; it does not train, run DPO, run GT-only SFT, or perform an optimizer step.

## Counts

- Step30 closer to pseudo target: `0`
- Step30 closer to V_bg: `0`
- Bad teacher learned count: `0`
- Loss/path bug suspect count: `48`

## Movement Labels

- `DOES_NOT_LEARN_TARGET`: `48`

## Mean Direction Metrics

- Step30-to-pseudo full L1 delta vs Step0: `0.005742` (negative means closer to pseudo)
- Step30-to-V_bg full L1 delta vs Step0: `0.011993` (negative means closer to V_bg)
- Step30-to-V_bg mask L1 delta vs Step0: `0.017009`
- Step30-to-V_bg outside L1 delta vs Step0: `0.012223`
- cosine(train, teacher) full: `0.262656`
- cosine(train, V_bg) full: `-0.140454`
- cosine(teacher, V_bg) full: `0.025774`

Interpretation: if Step30 moves closer to the pseudo target while moving farther from `V_bg`, Exp46 likely learned a bad/global-drift teacher. If Step30 does not move toward the pseudo target, the remaining suspicion would shift to loss/target path implementation.
