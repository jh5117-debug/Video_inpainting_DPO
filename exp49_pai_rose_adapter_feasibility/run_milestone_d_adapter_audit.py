#!/usr/bin/env python3
"""Generate Exp49 Milestone D ROSE adapter-feasibility audit reports."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path("/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp49_rose_adapter")
ROSE_ROOT = Path("/mnt/nas/hj/H20_Video_inpainting_DPO/third_party/ROSE/Kunbyte-AI_ROSE")
REPORT_DIR = PROJECT_ROOT / "reports"
REGISTRY_DIR = PROJECT_ROOT / "experiment_registry/exp49_pai_rose_adapter_feasibility"
PRD_MAIN = PROJECT_ROOT / "PRD/00_current_status.md"
PRD_MATRIX = PROJECT_ROOT / "PRD/01_experiment_matrix.md"
PRD_EXP49 = PROJECT_ROOT / "PRD/46_exp49_pai_rose_adapter_feasibility.md"


def run(cmd: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def grep(pattern: str) -> list[str]:
    cmd = ["grep", "-R", "-n", "-E", pattern, "--include=*.py", "."]
    proc = subprocess.run(cmd, cwd=ROSE_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return [line for line in proc.stdout.splitlines() if line.strip()]


def append_once(path: Path, marker: str, block: str) -> None:
    text = path.read_text()
    if marker not in text:
        path.write_text(text.rstrip() + "\n\n" + block.strip() + "\n")


def replace_current_status(path: Path, status: str, body: str) -> None:
    path.write_text(f"# Exp49 Status\n\nCurrent status: `{status}`\n\n{body.strip()}\n")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    generated = datetime.now(timezone.utc).astimezone().isoformat()
    host = socket.gethostname()
    branch = run(["git", "branch", "--show-current"], PROJECT_ROOT)
    commit = run(["git", "rev-parse", "HEAD"], PROJECT_ROOT)
    rose_commit = run(["git", "rev-parse", "HEAD"], ROSE_ROOT)

    files = {
        "README.md": ROSE_ROOT / "README.md",
        "inference.py": ROSE_ROOT / "inference.py",
        "wan_transformer3d.py": ROSE_ROOT / "rose/models/wan_transformer3d.py",
        "pipeline_wan_fun_inpaint.py": ROSE_ROOT / "rose/pipeline/pipeline_wan_fun_inpaint.py",
        "dataset_image_video.py": ROSE_ROOT / "rose/data/dataset_image_video.py",
        "diff_mask_predictor.py": ROSE_ROOT / "rose/models/diff_mask_predictor.py",
        "lora_utils.py": ROSE_ROOT / "rose/utils/lora_utils.py",
        "wan_civitai.yaml": ROSE_ROOT / "configs/wan2.1/wan_civitai.yaml",
    }
    file_hashes = {name: sha256(path) for name, path in files.items() if path.exists()}

    train_script_candidates = [
        str(p.relative_to(ROSE_ROOT))
        for p in ROSE_ROOT.rglob("*")
        if p.is_file()
        and any(token in p.name.lower() for token in ["train", "finetune", "dpo", "adapter"])
    ]
    optimizer_hits = grep(r"(accelerator\.backward|optimizer\.step|F\.mse_loss|mse_loss|train_dataloader|DataLoader|save_state|checkpoint)")
    lora_hits = grep(r"(LoRANetwork|prepare_optimizer_params|save_weights|load_weights|merge_lora|WanTransformer3DModel)")
    no_grad_hits = grep(r"(@torch\.no_grad|with torch\.no_grad)")
    diffmask_hits = grep(r"(DiffMaskPredictor|diff_mask)")

    status = "ROSE_TRAINING_FORWARD_BLOCKED"
    decision = {
        "official_training_script_present": False,
        "official_loss_present": False,
        "inference_pipeline_no_grad": True,
        "transformer_forward_differentiable": True,
        "lora_utility_present": True,
        "checkpoint_load_present": True,
        "lora_save_reload_present": True,
        "target_parameterization_explicit": False,
        "target_parameterization_inferred": "FlowMatchEulerDiscreteScheduler; transformer output is passed as noise_pred/flow residual to scheduler.step, but official training target/loss is not released.",
        "adapter_feasibility": status,
    }

    rows = [
        {
            "component": "model_family",
            "file": "README.md; configs/wan2.1/wan_civitai.yaml",
            "finding": "Wan2.1-Fun-1.3B-InP based video inpainting diffusion transformer; README says training trains WanTransformer3D and freezes other parts.",
            "adapter_impact": "Backbone is conceptually adapter-suitable, but released code does not include executable training recipe.",
        },
        {
            "component": "inference_entrypoint",
            "file": "inference.py:74-155",
            "finding": "Only CLI entrypoint is inference-oriented; loads base, transformer, scheduler, VAE, text/CLIP encoders and runs with torch.no_grad.",
            "adapter_impact": "Usable for baseline smoke after path wrapper/symlinks; not a training entrypoint.",
        },
        {
            "component": "pipeline",
            "file": "rose/pipeline/pipeline_wan_fun_inpaint.py:468-729",
            "finding": "WanFunInpaintPipeline.__call__ is decorated with @torch.no_grad and runs denoising loop through scheduler.step.",
            "adapter_impact": "Official pipeline must not be used directly for gradient training without an isolated wrapper.",
        },
        {
            "component": "target_parameterization",
            "file": "rose/pipeline/pipeline_wan_fun_inpaint.py:557-699; configs/wan2.1/wan_civitai.yaml",
            "finding": "FlowMatchEulerDiscreteScheduler with num_train_timesteps=1000, shift=5.0; transformer prediction is consumed by scheduler.step as noise_pred.",
            "adapter_impact": "Training target is not explicitly released; likely flow/velocity style residual must be reconstructed before any LoVI-DPO gate.",
        },
        {
            "component": "trainable_forward",
            "file": "rose/models/wan_transformer3d.py:810-1026",
            "finding": "WanTransformer3DModel.forward is ordinary PyTorch forward and supports gradient checkpointing when grad is enabled.",
            "adapter_impact": "A custom Exp49 wrapper can test gradients later, but Milestone D does not prove optimizer training.",
        },
        {
            "component": "trainable_modules",
            "file": "README.md:226-240; rose/utils/lora_utils.py:158-366",
            "finding": "README says only WanTransformer3D is trained; LoRANetwork targets WanTransformer3DModel and can prepare optimizer params.",
            "adapter_impact": "Trainable scope is plausible for transformer LoRA/full transformer, but no official training script applies it.",
        },
        {
            "component": "frozen_modules",
            "file": "inference.py:99-134",
            "finding": "Tokenizer, T5 text encoder, CLIP image encoder, VAE, scheduler, transformer are assembled; README says non-transformer parts are frozen during training.",
            "adapter_impact": "Reference/frozen split is feasible in principle but needs Exp49 wrapper proof.",
        },
        {
            "component": "checkpoint_load",
            "file": "rose/models/wan_transformer3d.py:1080-1192",
            "finding": "WanTransformer3DModel.from_pretrained loads config and safetensors/bin weights.",
            "adapter_impact": "Strict reload for base transformer appears feasible.",
        },
        {
            "component": "adapter_save_reload",
            "file": "rose/utils/lora_utils.py:275-339",
            "finding": "LoRANetwork has load_weights and save_weights for safetensors/torch state_dict.",
            "adapter_impact": "LoRA save/reload is feasible if a custom training wrapper reaches one-step gate.",
        },
        {
            "component": "dataset_loader",
            "file": "rose/data/dataset_image_video.py:186-362; 365-589",
            "finding": "ImageVideoDataset and ImageVideoControlDataset read csv/json annotations and synthesize random inpaint masks.",
            "adapter_impact": "Native dataset does not directly consume VOR-OR manifests; an Exp49 adapter dataset would be required.",
        },
        {
            "component": "mask_polarity",
            "file": "rose/pipeline/pipeline_wan_fun_inpaint.py:592-642; rose/data/dataset_image_video.py:347-351",
            "finding": "Dataset uses mask=1 as masked region; inference pipeline binarizes mask_video and then builds latent mask from 1 - mask_condition.",
            "adapter_impact": "VOR mask polarity must be tested in inference smoke before any training gate.",
        },
        {
            "component": "difference_mask_predictor",
            "file": "rose/models/diff_mask_predictor.py:6-42",
            "finding": "DiffMaskPredictor class exists, but grep found no integration call outside its own definition.",
            "adapter_impact": "Side-effect predictor is not an exposed training/inference hook in released code.",
        },
        {
            "component": "official_training_release",
            "file": "repo tree + grep",
            "finding": f"No train/finetune script or optimizer/backward loop found. Candidates: {train_script_candidates or 'none'}",
            "adapter_impact": "Blocks ROSE_TRUE_ADAPTER_FEASIBLE status in Milestone D.",
        },
    ]

    csv_path = REPORT_DIR / "exp49_rose_code_adapter_feasibility_audit.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["component", "file", "finding", "adapter_impact"])
        writer.writeheader()
        writer.writerows(rows)

    inventory = {
        "generated": generated,
        "hostname": host,
        "branch": branch,
        "commit": commit,
        "rose_repo": str(ROSE_ROOT),
        "rose_commit": rose_commit,
        "status": status,
        "file_hashes": file_hashes,
        "decision": decision,
        "counts": {
            "train_script_candidates": len(train_script_candidates),
            "optimizer_backward_hits": len(optimizer_hits),
            "lora_hits": len(lora_hits),
            "no_grad_hits": len(no_grad_hits),
            "diffmask_hits": len(diffmask_hits),
        },
        "train_script_candidates": train_script_candidates,
        "optimizer_backward_hits": optimizer_hits[:80],
        "no_grad_hits": no_grad_hits,
        "diffmask_hits": diffmask_hits,
    }
    json_path = REPORT_DIR / "exp49_rose_training_forward_inventory.json"
    json_path.write_text(json.dumps(inventory, indent=2))

    md = f"""# Exp49 ROSE Code / Adapter Feasibility Audit

Status: `{status}`

Generated: {generated}
Host: `{host}`
Branch: `{branch}`
Commit: `{commit}`
ROSE repo: `{ROSE_ROOT}`
ROSE commit: `{rose_commit}`

## Decision

ROSE is not promoted to `ROSE_TRUE_ADAPTER_FEASIBLE` in Milestone D. The released official repo contains an inference entrypoint, a differentiable Wan transformer, dataset classes, and LoRA utilities, but it does not release an executable training script, optimizer/backward loop, explicit loss, or explicit training target construction. Therefore this milestone marks adapter feasibility as `ROSE_TRAINING_FORWARD_BLOCKED`.

This does not mean ROSE is inference-only forever. It means a future Exp49 wrapper would first need to reconstruct the ROSE-native FlowMatch target and prove zero-gap / one-step / strict reload before claiming adapter feasibility.

## Architecture Table

| Component | File | Finding | Adapter impact |
| --- | --- | --- | --- |
"""
    for row in rows:
        md += f"| `{row['component']}` | `{row['file']}` | {row['finding']} | {row['adapter_impact']} |\n"

    md += f"""
## Key Evidence

- `inference.py` hard-codes local base/transformer roots (`models/Wan2.1-Fun-1.3B-InP`, `weights/transformer`) and runs validation samples inside `with torch.no_grad()`.
- `WanFunInpaintPipeline.__call__` is decorated with `@torch.no_grad()`, so the official pipeline cannot be directly reused as a training forward.
- `WanTransformer3DModel.forward()` is a normal PyTorch forward and can run with gradients when called outside the no-grad pipeline.
- `LoRANetwork` supports `WanTransformer3DModel`, `prepare_optimizer_params`, `save_weights`, and `load_weights`.
- `DiffMaskPredictor` is defined but not integrated in the released inference/training path.
- No official train/finetune script, `accelerator.backward`, or `optimizer.step` loop was found in the released repo.

## Gate Consequence

- Milestone E official inference remains blocked by `ROSE_ENV_PARTIAL` until the user accepts Python 3.10 as a practical env or a true Python 3.12 Torch env is created.
- Milestone G trainable forward / zero-gap / one-step must not run yet because `ROSE_TRUE_ADAPTER_FEASIBLE` was not reached.

## Safety

No inference, training, optimizer step, checkpoint update, VOR-Eval use, H20 action, or official ROSE source modification was performed.
"""
    (REPORT_DIR / "exp49_rose_code_adapter_feasibility_audit.md").write_text(md)

    registry_body = """Milestone D audited the released ROSE official code. The repo exposes a differentiable WanTransformer3DModel and LoRA save/load utilities, but no executable official training script, optimizer/backward loop, explicit loss, or explicit FlowMatch training target construction was found. No inference/training was run."""
    replace_current_status(REGISTRY_DIR / "status.md", status, registry_body)

    with (REGISTRY_DIR / "results.tsv").open("a") as f:
        f.write(f"D_adapter_feasibility\t{status}\tNo official train/finetune optimizer/backward/loss script found; transformer forward and LoRA utilities present.\n")

    (REGISTRY_DIR / "metric_summary.md").write_text(
        "# Exp49 Metric Summary\n\n"
        "Milestone D computed no video metrics. It is a code and adapter-feasibility audit only.\n\n"
        f"Current status: `{status}`.\n"
    )
    (REGISTRY_DIR / "qualitative_summary.md").write_text(
        "# Exp49 Qualitative Summary\n\n"
        "Milestone D performed no visual promotion. ROSE remains untested on VOR-OR outputs in this milestone.\n\n"
        "Inference and training claims remain forbidden until the relevant gates run.\n"
    )
    readme_path = REGISTRY_DIR / "README.md"
    append_once(
        readme_path,
        "## Milestone D",
        f"""## Milestone D

Status: `{status}`.

Official ROSE code exposes a differentiable WanTransformer3DModel and LoRA utilities, but released code does not expose a complete training loop/loss/target. See `reports/exp49_rose_code_adapter_feasibility_audit.md`.
""",
    )
    config_path = REGISTRY_DIR / "config.yaml"
    append_once(
        config_path,
        "milestone_d:",
        f"""milestone_d:
  status: {status}
  rose_repo: {ROSE_ROOT}
  rose_commit: {rose_commit}
  no_training_run: true
  no_optimizer_step: true
""",
    )
    paths_path = REGISTRY_DIR / "paths.yaml"
    append_once(
        paths_path,
        "milestone_d_reports:",
        """milestone_d_reports:
  audit_md: reports/exp49_rose_code_adapter_feasibility_audit.md
  audit_csv: reports/exp49_rose_code_adapter_feasibility_audit.csv
  inventory_json: reports/exp49_rose_training_forward_inventory.json
""",
    )

    prd_block = f"""## 2026-06-30 Exp49 ROSE Adapter Feasibility Audit

Exp49 Milestone D audited official ROSE code on PAI. Status: `{status}`. The released code has inference, dataset loaders, a differentiable WanTransformer3DModel, and LoRA utilities, but no executable official training script, optimizer/backward loop, explicit loss, or explicit FlowMatch training target construction was found. No inference, training, optimizer step, checkpoint update, VOR-Eval use, H20 action, or official ROSE source modification was performed.
"""
    append_once(PRD_MAIN, "2026-06-30 Exp49 ROSE Adapter Feasibility Audit", prd_block)
    append_once(
        PRD_MATRIX,
        "2026-06-30 Exp49 Feasibility Update",
        f"""## 2026-06-30 Exp49 Feasibility Update

| Experiment | Milestone | Status | Notes |
| --- | --- | --- | --- |
| `exp49_pai_rose_adapter_feasibility` | D adapter audit | `{status}` | Official code exposes transformer forward and LoRA utilities, but no released training loop/loss/target; no inference/training run. |
""",
    )
    append_once(
        PRD_EXP49,
        "## Milestone D Update - 2026-06-30",
        f"""## Milestone D Update - 2026-06-30

Status: `{status}`.

Official ROSE code was audited on PAI. `inference.py` and `WanFunInpaintPipeline.__call__` are no-grad inference paths. `WanTransformer3DModel.forward()` is differentiable and `rose/utils/lora_utils.py` exposes LoRA optimizer-param and save/load helpers, but the released repository does not include an executable training script, optimizer/backward loop, explicit loss, or explicit FlowMatch target construction. ROSE is therefore not yet a proven adapter candidate; a future isolated wrapper would need zero-gap / one-step proof before any adapter claim.

No inference, training, optimizer step, checkpoint update, VOR-Eval use, H20 action, or official ROSE source modification was performed.
""",
    )

    print(json.dumps({"status": status, "md": str(REPORT_DIR / "exp49_rose_code_adapter_feasibility_audit.md")}, indent=2))


if __name__ == "__main__":
    main()
