# Exp50 VOID Preference-Wrapper Blocker

Status: `VOID_PREFERENCE_WRAPPER_REQUIRED_CONFIRMED`

The previous G1 gate was correctly blocked by `VOID_TRAINABLE_FORWARD_BLOCKED_PREFERENCE_WRAPPER_REQUIRED`.

## Exact Blocker

Official VOID training has a valid SFT data path and can load the Exp50 train4 data in bucket mode. However, the requested LoVI-DPO style gate needs:

- policy and frozen reference transformers;
- winner and loser target latents from the same source;
- identical condition, prompt, quadmask, noise, timestep, and scheduler state;
- winner/reference and loser/reference loss gaps;
- preference margin and DPO-style objective;
- reference gradients exactly zero;
- policy gradients finite.

The official script does not provide this interface. Therefore an isolated wrapper under `exp50_pai_void_adapter_feasibility/` is required.

## Deepspeed Position

Deepspeed is not required to begin a tiny single-process wrapper forward if GPU memory is sufficient. It remains intentionally uninstalled. A controlled deepspeed install is only allowed if the single-process wrapper is proven impossible and the wheelhouse is extended without torch/CUDA drift.

## Next Gate

H1 must first prove SFT-forward parity or code-equivalence against official target construction. H2 may then implement policy/reference preference forward. H3/H4/H5 remain locked behind those gates.
