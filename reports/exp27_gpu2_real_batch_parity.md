# Exp27 GPU2 Real-Batch Parity

Date: 2026-06-23 UTC

Runtime root:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260623`

## Scope

This gate exercised Exp27 SDPO and Linear-DPO objective plumbing on GPU tensors
shaped like real DiffuEraser epsilon batches. It did not start any DPO
training or long study.

## SDPO Real-Batch Parity

Output:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260623/exp27_sdpo_real_batch_parity/real_batch_sdpo_parity.json`

Result:

- Status: `passed`.
- Device: `cuda`.
- dtype: `torch.bfloat16`.
- Objective finite: `true`.
- Objective: `0.010575979948043823`.
- Grad norm: `0.008771423250436783`.
- Safe lambda min/max: `1.0 / 1.0`.

## Linear-DPO Frozen / EMA Real-Batch Parity

Output:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/overnight_exp25_26_27_20260623/exp27_linear_real_batch_parity/real_batch_linear_parity.json`

Result:

- Status: `passed`.
- Device: `cuda`.
- dtype: `torch.bfloat16`.
- Loss finite: `true`.
- Loss: `-0.04078614339232445`.
- Grad norm: `0.5717411041259766`.
- Ratio min/max: `0.009999999776482582 / 0.9900000095367432`.
- EMA max absolute difference: `0.0`.

Next required work remains faithful LocalDPO pair construction and any
micro-training gates. No long training was launched.

