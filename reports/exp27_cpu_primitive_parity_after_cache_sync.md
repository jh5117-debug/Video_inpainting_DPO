# Exp27 CPU Primitive Parity After Cache Sync

Date: 2026-06-23

PAI output:

`/mnt/nas/hj/H20_Video_inpainting_DPO/logs/autoresearch/exp27_paper_grounded_preference_study/pai_cpu_parity_cache_sync_latest`

Status:

`PASSED`

## Results

| gate | status | key number |
| --- | --- | --- |
| LocalDPO official mask | passed | shape `[13, 120, 216]`, sha256 `6560a6bb7100bb22622edc3962df7b678eaef6aed46b8d21749573405b662041` |
| LocalDPO latent fusion | passed | outside preservation max abs `0.0` |
| Diffusion-SDPO lambda | passed | max abs diff `0.0` |
| Linear-DPO primitive | passed | loss `0.09172474592924118`, grad finite |
| Linear-DPO EMA | passed | max abs diff `0.0` |

## Compatibility Note

Pinned Local-DPO official code expects an RGB byte buffer from a call named
`tostring_argb()` and also references `cv2` without importing it. Exp27 now
installs a narrow runtime compatibility shim while keeping the official cache
read-only. This is documented as official algorithm parity under a runtime
compatibility layer, not a modified official repository.

## Not Yet Passed

`real_diffueraser_batch_parity` remains pending. No GPU batch was launched in
this CPU command, so no Data Study, Objective Study, or RC-FPO run is allowed
yet.
