# Exp19 Boundary-Gated Flow-Adapter DPO

Exp19 tests whether explicit completed-flow features can improve the current
best DiffuEraser DPO checkpoint without changing the Exp11 loss.

Current scope:

- start from Exp11 outer b0.75 S2 Stage2 weights
- freeze the base DiffuEraser Stage2 model
- train only zero-initialized flow adapter parameters
- keep the Exp11 region-local normalized-gap DPO objective unchanged
- run only limit=100 flow-cache / Stage2-adapter gates before any expansion

Do not use this folder to modify old Exp11/Exp18 code or shared training logic.
