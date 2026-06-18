# Exp19 Architecture Preflight

- UNetMotionModel.forward has down residual argument: `True`
- UNetMotionModel.forward has mid residual argument: `True`
- detected shared down-residual double-add risk: `True`
- allow_mid_only: `False`
- status: `BLOCKED_MULTI_SCALE_INJECTION_UNSAFE`

## Decision

Do not launch Exp19 training from this shared UNetMotionModel path.

Reason: the requested multi-scale flow adapter needs clean additive
down/mid residual semantics. The current shared forward has a
ControlNet-style branch and an unconditional second
`down_block_additional_residuals` addition. Passing both down and
mid residuals would double-add the down residuals. Passing only down
residuals falls into the legacy T2I-adapter path with a different
shape contract. Reducing Exp19 to mid-block-only would violate the
requested method definition.

Allowed next implementation path: copy `libs/unet_motion_model.py`
into `exp19_boundary_gated_flow_adapter_dpo/code/`, implement a
clean Exp19-only residual interface there, and write a matching
Exp19 inference wrapper. Until that exists, training is blocked.
