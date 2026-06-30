# Exp50 VOID One-Step Gate

Status: `VOID_TRAINABLE_FORWARD_BLOCKED`

One-step was not run because the zero-gap/trainable-forward gate is blocked. The official VOID training path can load data, but it does not provide the required reference-model/winner-loser preference interface. A one-step SFT run would be off-protocol for this gate and could create misleading evidence.

Next minimal action: design an isolated Exp50 VOID-native preference-forward wrapper, or perform a controlled deepspeed wheelhouse install only if choosing the official SFT training path deliberately. Do not run 10-step until a true one-step gate passes.
