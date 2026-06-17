# Exp18 PAI Run Blocked Report

Date: 2026-06-17

Attempted from HAL:

```text
ssh pai
ssh PAI
test /mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO
test /mnt/workspace/hj/nas_hj/data/external/davis_432_240
```

Observed:

```text
ssh: Could not resolve hostname pai
/mnt/workspace paths not visible in this HAL session
```

Conclusion:

```text
Exp18 implementation is prepared locally, but real cache/training/eval must be
run from a PAI session or a HAL session with PAI mount/ssh configured.
```

No Exp18 training was launched.

