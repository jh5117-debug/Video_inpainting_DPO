# Exp59 Qualitative Summary

Exp58B visually reviewed all 8 Kubric Gate8 pages. Exp59 then ran official VOID pass1 inference and opened all 8 official-output review contact sheets.

Qualitative result:

- output technically valid: 8/8
- outside/background stable or safe: 8/8
- `target_hit=false`: 8/8
- medium-hard loser diagnostics: 2/8
- too-close/weak diagnostics: 2/8
- transition residual/damage: 6/8

The native data is valid for official inference diagnostics and same-model loser-generation smoke. It is not adapter evidence and is not ready for one-step training because all rows carry the `target_hit=false` caveat and transition-region residuals remain common.

Against Exp50 VOR-derived Gate8, Kubric-native Gate8 is not stronger as a usable-loser pool: VOR had 6/8 usable or bounded-loser outputs, while this Kubric Gate8 has only 2/8 medium-hard loser diagnostics and 2/8 too-close/weak diagnostics. Data mismatch remains suspected but unconfirmed because the native data target quality is weak.
