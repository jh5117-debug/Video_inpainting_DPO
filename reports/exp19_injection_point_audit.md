# Exp19 Injection Point Audit

Status:

```text
BLOCKED_MULTI_SCALE_INJECTION_UNSAFE
```

Candidate injection points:

| Site | Shared-code support | Exp19 status |
|---|---|---|
| mid block | `mid_block_additional_residual` exists | safe for mid-only debugging |
| down skip residuals | `down_block_additional_residuals` exists | unsafe with mid residual because current forward double-adds |
| down intrablock residuals | `down_intrablock_additional_residuals` exists | different T2I-adapter contract, not same multi-scale skip list |
| up blocks | no clean public residual interface | would require copied model/wrapper |

Conclusion: the requested zero-initialized multi-scale flow adapter should not
be trained through the shared `UNetMotionModel` interface.
