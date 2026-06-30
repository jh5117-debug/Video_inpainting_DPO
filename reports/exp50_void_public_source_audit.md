# Exp50 VOID Public Source Audit

## Public Sources

| Source | URL | Key facts | Exp50 implication |
| --- | --- | --- | --- |
| Official GitHub | https://github.com/Netflix/void-model | Apache-2.0, ECCV 2026 accepted, repo includes `VLM-MASK-REASONER/`, `data_generation/`, `datasets/`, `inference/`, `sample/`, `scripts/cogvideox_fun/`, `videox_fun/`, and training scripts. | Stronger than ROSE for adapter feasibility because training/data-generation entry points are public. |
| Project page | https://void-model.github.io/ | VOID targets physically plausible object and interaction deletion using affected-region reasoning and video diffusion. | We must report interaction deletion separately from ordinary VOR-OR. |
| HF model | https://huggingface.co/netflix/void-model | Apache-2.0; Pass1 required, Pass2 optional warped-noise refinement; built on CogVideoX-Fun-V1.5-5b-InP; input video + quadmask + prompt; default 384x672, max 197 frames. | Download pass1/pass2; start with Pass1 inference and only use Pass2 if present. |
| Base model | https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.5-5b-InP | CogVideoX-Fun V1.5 5B InP base, model card lists about 20GB storage. | Required base model under `weights/void`. |

## Quadmask Semantics

- `0`: primary object to remove.
- `63`: overlap of primary and affected regions.
- `127`: affected / interaction region.
- `255`: background keep.

## Training Scripts From Public README

- Pass1: `bash scripts/cogvideox_fun/train_void.sh`.
- Pass2: `bash scripts/cogvideox_fun/train_void_warped_noise.sh` with `TRANSFORMER_PATH` set to a Pass1 checkpoint.
- Training uses `--train_mode="void"`, `--use_quadmask`, `--use_vae_mask`, `--train_data_meta`, `--pretrained_model_name_or_path`, and checkpoint/output parameters.
- Public README says training was run on 8x A100 80GB with DeepSpeed ZeRO stage 2.

## Data Generation

- HUMOTO path requires external HUMOTO access, Blender, Mixamo Remy/Sophie assets, and optional textures.
- Kubric path can generate object-only interactions and fetches assets from Google Cloud Storage; do not download HUMOTO unless authorized.
- For Exp50, VOR-Train conversion avoids Gemini/SAM/SAM2 for first gate because VOR already provides object masks and `V_bg` for affected-map construction.

## Risk Register

- NAS target directories for assets are not writable by `hj` at Milestone A start.
- Base model and 5B inference are heavy; need disk and GPU checks before download/inference.
- Inference success does not imply adapter success.
- VOID interaction deletion may not align perfectly with VOR-OR object-removal metrics; report separately.
