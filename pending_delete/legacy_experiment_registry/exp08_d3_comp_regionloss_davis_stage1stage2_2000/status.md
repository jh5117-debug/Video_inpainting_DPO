# Status

Prepared for PAI manual launch.

Do not launch unless these prechecks pass:

- D3 comp PAI manifest exists and contains no `/home/nvme01` paths.
- SFT-48000 DiffuEraser weights exist.
- ProPainter weights exist.
- DAVIS validation directory exists with frames and masks.
- `LOSS_REGION_MODE=region` is implemented in both Stage1 and Stage2 training.
- DPO diagnostics are enabled.

H20 is not used for this experiment while small-mask D2 data generation is running.
