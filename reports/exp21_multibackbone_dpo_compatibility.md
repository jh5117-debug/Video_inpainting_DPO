# Exp21 multibackbone DPO compatibility

| model | backend_path | native_target | policy_forward | reference_forward | zero_init_parity | finite_dpo_loss | finite_grad | save_reload | inference_with_adapter | status | blocker | next_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DiffuEraser | exp21_multibackbone_videodpo_br_smoke/backends/diffueraser | epsilon/noise residual as used by project trainer | existing backend; wrapper pending | existing frozen ref; wrapper pending | pending | pending | pending | pending | pending | PENDING_REAL_SMOKE |  | wrap existing DiffuEraser trainer interface without changing shared code |
| ProPainter |  |  |  |  |  |  |  |  |  | NOT_APPLICABLE_NON_DIFFUSION | non-diffusion propagation baseline |  |
| EffectErase |  |  |  |  |  |  |  |  |  | WAITING_AUTH | VOR data authorization pending; code/weight readiness only |  |
| FloED |  |  |  |  |  |  |  |  |  | PENDING_AUDIT | training forward / public checkpoint not yet verified |  |
| CoCoCo |  |  |  |  |  |  |  |  |  | PENDING_AUDIT | SD inpainting dependency and trainable modules need isolated env |  |
| VideoComposer |  |  |  |  |  |  |  |  |  | PENDING_AUDIT | exact repo/checkpoint/inference entry must be disambiguated |  |
| VACE |  |  |  |  |  |  |  |  |  | PENDING_AUDIT | Wan/VACE flow-matching forward and LoRA path need audit |  |
| MiniMax-Remover |  |  |  |  |  |  |  |  |  | PENDING_AUDIT | independent env and transformer forward need audit |  |
| VideoPainter |  |  |  |  |  |  |  |  |  | PENDING_AUDIT | Exp14 adapter was negative; new smoke only, no quality claim |  |
