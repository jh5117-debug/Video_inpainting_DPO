# H20 Experiment Registry Audit

Updated: 2026-06-04

## SSH Status

HAL attempted H20 SSH using:

```text
ssh -i ~/.ssh/codex_h20_2 ubuntu@27.190.15.128
```

The connection failed in this turn:

```text
kex_exchange_identification: read: Connection reset by peer
Connection reset by 27.190.15.128 port 22
ssh: connect to host 27.190.15.128 port 22: Connection timed out
```

No H20 commands were run after the connection failure. No H20 jobs were killed,
started, or modified.

## Consequence

The H20 scan requested by the user remains blocked for this turn. Existing local
registry entries still include previously recorded H20 paths for New Exp6,
Exp9-nocomp, and Exp9 no-lose, but fresh H20 validation must be retried when SSH
is available.

## H20 Items That Must Be Rechecked

- `/home/nvme01/H20_Video_inpainting_DPO/weights/diffuEraser/converted_weights_step48000`
- Exp7 run dirs and `dpo_diagnostics.csv`
- Exp9 nocomp/no-lose run dirs and target eval outputs
- GPU availability before any H20 gate launch
- Whether H20 has or needs local `exp07_fix_videodpo_smallmask15_20_prior_k4` data

## Retry Update

A second short-timeout SSH attempt also failed:

```text
ssh: connect to host 27.190.15.128 port 22: Connection timed out
```

Therefore no H20 Exp7-fix no-lose gate was launched in this turn.
