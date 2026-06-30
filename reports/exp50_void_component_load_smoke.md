# Exp50 VOID Component Load Smoke

Timestamp: 2026-06-30T16:50:03+08:00

Status: `VOID_COMPONENT_LOAD_PASS`

## Scope

This F0 smoke read official configs, tokenizer/scheduler metadata, safetensors headers, key prefixes, and optional meta-device model skeletons. It did not run inference, did not train, did not step an optimizer, and did not load the full 5B model onto GPU.

## Assets

- VOID weights: `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/void/netflix_void-model`
- Base model: `/mnt/nas/hj/H20_Video_inpainting_DPO/weights/void/CogVideoX-Fun-V1.5-5b-InP`
- Official repo: `/mnt/nas/hj/H20_Video_inpainting_DPO/third_party/VOID/Netflix_void-model`
- Env: `/home/hj/conda_envs/void_exp50_official_v2`

## Config And Metadata

- Required files checked: 24
- Blocking missing/fail checks: 0
- Tokenizer config load: `PASS`
- Scheduler config load: `PASS`
- Transformer meta init: `PASS`
- VAE meta init: `PASS`

## Safetensors Header Read

- Pass1: `HEADER_OK`, keys 1024, disk bytes 11143042384
- Pass2: `HEADER_OK`, keys 1024, disk bytes 11143042384
- Base transformer: `HEADER_OK`, keys 1024, disk bytes 11142305104
- Base VAE: `HEADER_OK`, keys 436, disk bytes 431221142

## Memory Estimate

- Total asset disk bytes: 43385161698
- Base weight disk bytes: 21098174830
- VOID Pass1+Pass2 disk bytes: 22286084768
- Full GPU load attempted: False
- Reason: skipped; F0 is no-inference lightweight metadata smoke and full 5B load is not required before F1

## Decision

`VOID_COMPONENT_LOAD_PASS`. F1 official sample inference can proceed only if official sample data exists; otherwise F2 may proceed under the requested `VOID_OFFICIAL_SAMPLE_NOT_PROVIDED with VOID_COMPONENT_LOAD_PASS` rule.
