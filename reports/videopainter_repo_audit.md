# VideoPainter Repo Audit

Status: repo found.

## Local Repo

```text
/home/hj/dpo-2-1-exp/third_party_baselines/VideoPainter
```

Remote:

```text
https://github.com/TencentARC/VideoPainter.git
```

Commit checked on HAL:

```text
bbab6cd5cd5cb89f0e2444305c32fd74a010ae0a
```

Branch:

```text
main
```

## License

License file:

```text
/home/hj/dpo-2-1-exp/third_party_baselines/VideoPainter/LICENSE
```

The license permits academic / research / education use for VideoPainter and
also includes CogVideoX license restrictions for the CogVideoX 5B base model.

## Repo Contents

Found:

- `train/`
- `infer/`
- `evaluate/`
- `diffusers/`
- `app/`
- `data_utils/`
- `README.md`
- `LICENSE`

## Pretrained Weights

README points to:

- `TencentARC/VideoPainter` checkpoints
- `THUDM/CogVideoX-5b-I2V`
- optional `FLUX.1-Fill-dev`

HAL cache also contains:

```text
/home/hj/.cache/huggingface/hub/models--TencentARC--VideoPainter
```

## Data Support

The README documents VPBench / VPData and includes a reprocessed DAVIS layout:

```text
data/davis/JPEGImages_432_240
data/davis/test_masks
data/davis/davis_caption
```

Training examples use VPData / VideoVO / Pexels CSV metadata and raw-video
folders rather than the current D3 JSONL manifest. A conversion layer is needed.

## Decision

Repo is usable for feasibility. It has training, inference, evaluation, config,
checkpoint, and dataset documentation.

