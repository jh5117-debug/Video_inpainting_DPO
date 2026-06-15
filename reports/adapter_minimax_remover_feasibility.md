# MiniMax-Remover Adapter Feasibility

Date: 2026-06-15

Status: feasibility only. No training launched.

## Sources Checked

Project-local files:

```text
/home/hj/H20_Video_inpainting_DPO_hal/DPO_finetune/infer_minimax_candidate.py
/home/hj/H20_Video_inpainting_DPO_hal/weights/minimax_remover/.gitkeep
/home/hj/.cache/huggingface/hub/models--zibojia--minimax-remover
```

GitHub:

```text
https://github.com/zibojia/MiniMax-Remover
```

## Evidence

The public repository describes MiniMax-Remover as a two-stage video object remover and provides inference-facing files:

```text
gradio_demo/
pipeline_minimax_remover.py
transformer_minimax_remover.py
test_minimax_remover.py
requirements.txt
```

The README shows weight download and a quick-start inference example that loads:

```text
vae/
transformer/
scheduler/
```

Local project evidence is also inference-only. `DPO_finetune/infer_minimax_candidate.py` loads VAE / transformer / scheduler, runs `Minimax_Remover_Pipeline`, and saves output frames. It does not contain a train loop or DPO objective.

## Feasibility Checklist

| Question | Answer | Notes |
|---|---|---|
| Open source? | yes | Public GitHub repo. |
| Training code? | not verified | Repo listing exposes gradio/demo/test/pipeline, not train scripts. |
| Training data instructions? | not verified | No adapter-ready training-data workflow found locally. |
| Pretrained weights? | yes | HuggingFace model cache / README download path. |
| Diffusion / DiT based? | yes | Transformer + scheduler + VAE pipeline. |
| Can train on YouTubeVOS now? | no | No validated training entrypoint. |
| Can eval on DAVIS? | yes as frozen baseline | Existing inference wrapper can produce frames if weights are available. |
| Can define reference model? | not now | Needs train loop and duplicate model/reference integration. |
| Can connect DPO now? | no | No training entrypoint or loss path validated. |
| Current role | frozen baseline | Related-work / frozen baseline only. |

## Conclusion

Class: **C. frozen baseline only for now**.

MiniMax-Remover should not be described as a trainable DPO adapter at this stage. Although the paper/repo describe a trained two-stage remover, the public/local code currently available to this project is inference-oriented. Without verified training code and training data, adapter training would be speculative.

Allowed next step:

- Use MiniMax-Remover as a frozen baseline if weights and inference are stable.

Blocked next step:

- Do not start MiniMax DPO adapter training unless official train scripts and training data are found, copied into an isolated experiment folder, and pass a smoke test.
