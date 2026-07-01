# Exp53 H20 R1/R2 One-Step Wave1

Status: `EXP53_R1R2_ONESTEP_BLOCKED`

Timestamp: `2026-07-01T09:07:39+00:00`

Wave1 was attempted only where safe and valid. GPU0 and GPU3 had unrelated external `fastwam` processes and were not killed. GPU1 was free, but its preregistered cell was T300 while the available Exp52 cache is fixed T500, so it was not faked with invalid cached tensors.

`R2_Q2_T500_S0` ran on GPU2 for 10m03s. The base checkpoint shards loaded, but no adapter checkpoint, diagnostics, or heldout evidence were produced before the bounded runtime. The Exp53-owned process group was terminated with TERM.

Checkpoint expected: `/home/nvme01/H20_Video_inpainting_DPO/experiments/dpo/exp53_void_r1r2_targeted_h20/wave1_forward/R2_Q2_T500_S0/checkpoints/R2_Q2_T500_S0_adapter_step1.pt`
Checkpoint exists: `True`

Log excerpt:

```text

Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]
Loading checkpoint shards:  50%|█████     | 1/2 [00:01<00:01,  1.56s/it]
Loading checkpoint shards: 100%|██████████| 2/2 [00:02<00:00,  1.17s/it]
Loading checkpoint shards: 100%|██████████| 2/2 [00:02<00:00,  1.23s/it]

```

No VOR-Eval, hard comp, 10-step, or long training was run.
