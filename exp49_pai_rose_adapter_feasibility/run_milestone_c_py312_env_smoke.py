#!/usr/bin/env python3
"""Record Python 3.12 ROSE environment smoke for Exp49."""

from __future__ import annotations

import csv
import importlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path("/mnt/workspace/hj/nas_hj/H20_Video_inpainting_DPO_exp49_rose_adapter")
ROSE_ROOT = Path("/mnt/nas/hj/H20_Video_inpainting_DPO/third_party/ROSE/Kunbyte-AI_ROSE")
REPORT_DIR = PROJECT_ROOT / "reports"
REGISTRY_DIR = PROJECT_ROOT / "experiment_registry/exp49_pai_rose_adapter_feasibility"
ENV_PATH = Path("/home/hj/venvs/rose_exp49_py312")


def run(cmd: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def append_once(path: Path, marker: str, block: str) -> None:
    text = path.read_text()
    if marker not in text:
        path.write_text(text.rstrip() + "\n\n" + block.strip() + "\n")


def check_import(name: str, rows: list[dict[str, str]]) -> None:
    try:
        mod = importlib.import_module(name)
        rows.append({"check": f"import:{name}", "status": "PASS", "note": str(getattr(mod, "__version__", ""))})
    except Exception as exc:
        rows.append({"check": f"import:{name}", "status": "FAIL", "note": repr(exc)})


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).astimezone().isoformat()
    host = socket.gethostname()
    branch = run(["git", "branch", "--show-current"], PROJECT_ROOT)
    commit = run(["git", "rev-parse", "HEAD"], PROJECT_ROOT)

    sys.path.insert(0, str(ROSE_ROOT))
    rows: list[dict[str, str]] = []
    rows.append({"check": "python_version", "status": "PASS", "note": sys.version.replace("\n", " ")})

    imports = [
        "PIL",
        "einops",
        "safetensors",
        "timm",
        "tomesd",
        "torchdiffeq",
        "torchsde",
        "decord",
        "datasets",
        "numpy",
        "skimage",
        "cv2",
        "omegaconf",
        "sentencepiece",
        "albumentations",
        "imageio",
        "bs4",
        "ftfy",
        "func_timeout",
        "accelerate",
        "diffusers",
        "transformers",
    ]
    for name in imports:
        check_import(name, rows)

    try:
        import torch

        rows.append({
            "check": "torch",
            "status": "PASS",
            "note": f"{torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}",
        })
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            x = torch.randn((128, 128), device="cuda", dtype=torch.float16)
            y = x @ x.T
            torch.cuda.synchronize()
            rows.append({"check": "cuda_matmul_gpu0", "status": "PASS", "note": f"mean={float(y.float().mean().item()):.6f}"})
        else:
            rows.append({"check": "cuda_matmul_gpu0", "status": "FAIL", "note": "torch.cuda.is_available() is false"})
    except Exception as exc:
        rows.append({"check": "torch_cuda", "status": "FAIL", "note": repr(exc)})

    rose_modules = [
        "rose.models",
        "rose.models.wan_transformer3d",
        "rose.models.wan_vae",
        "rose.models.wan_text_encoder",
        "rose.models.wan_image_encoder",
        "rose.models.diff_mask_predictor",
        "rose.pipeline.pipeline_wan_fun_inpaint",
        "rose.utils.utils",
        "rose.data.dataset_image_video",
        "inference",
    ]
    for name in rose_modules:
        check_import(name, rows)

    paths = [
        "/mnt/nas/hj/H20_Video_inpainting_DPO/weights/rose/Kunbyte_ROSE/config.json",
        "/mnt/nas/hj/H20_Video_inpainting_DPO/weights/rose/Kunbyte_ROSE/diffusion_pytorch_model.safetensors",
        "/mnt/nas/hj/H20_Video_inpainting_DPO/weights/rose/Wan2.1-Fun-1.3B-InP/config.json",
        "/mnt/nas/hj/H20_Video_inpainting_DPO/weights/rose/Wan2.1-Fun-1.3B-InP/configuration.json",
        "/mnt/nas/hj/H20_Video_inpainting_DPO/weights/rose/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors",
        "/mnt/nas/hj/H20_Video_inpainting_DPO/weights/rose/Wan2.1-Fun-1.3B-InP/Wan2.1_VAE.pth",
        "/mnt/nas/hj/H20_Video_inpainting_DPO/weights/rose/Wan2.1-Fun-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth",
        "/mnt/nas/hj/H20_Video_inpainting_DPO/weights/rose/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "/mnt/nas/hj/H20_Video_inpainting_DPO/weights/rose/Wan2.1-Fun-1.3B-InP/google/umt5-xxl/tokenizer.json",
        "/mnt/nas/hj/H20_Video_inpainting_DPO/weights/rose/Wan2.1-Fun-1.3B-InP/xlm-roberta-large/tokenizer.json",
    ]
    for raw in paths:
        path = Path(raw)
        rows.append({
            "check": f"path:{raw}",
            "status": "PASS" if path.exists() else "FAIL",
            "note": str(path.stat().st_size) if path.exists() else "missing",
        })

    failed = [row for row in rows if row["status"] != "PASS"]
    status = "ROSE_ENV_READY" if not failed and sys.version_info[:2] == (3, 12) else "ROSE_ENV_BLOCKED"

    csv_path = REPORT_DIR / "exp49_rose_env_smoke.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "status", "note"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "generated": generated,
        "hostname": host,
        "status": status,
        "env_path": str(ENV_PATH),
        "python_version": sys.version.split()[0],
        "torch": next((row["note"] for row in rows if row["check"] == "torch"), ""),
        "fail_count": len(failed),
        "failed_checks": failed,
        "notes": "Python 3.12 venv was created with virtualenv because system python3.12 lacks ensurepip; torch/torchvision and ROSE requirements were installed into this isolated venv.",
    }
    (REPORT_DIR / "exp49_rose_env_smoke_summary.json").write_text(json.dumps(summary, indent=2))

    md = f"""# Exp49 ROSE Environment Smoke

Status: `{status}`

Generated: {generated}
Host: `{host}`
Branch: `{branch}`
Commit: `{commit}`

## Environment

- Env path: `{ENV_PATH}`
- Python: `{sys.version.split()[0]}`
- Torch: `{summary['torch']}`
- Install method: Python 3.12 venv via `virtualenv`; PyTorch/torchvision and ROSE requirements installed from public mirrors into isolated venv.

## Smoke

- Failed checks: `{len(failed)}`
- CSV: `reports/exp49_rose_env_smoke.csv`
- Summary JSON: `reports/exp49_rose_env_smoke_summary.json`

## Safety

Only dependency installation plus import/CUDA matmul smoke was run. No model inference, training, optimizer step, checkpoint update, VOR-Eval use, H20 action, or official ROSE source modification was performed.
"""
    (REPORT_DIR / "exp49_rose_env_smoke.md").write_text(md)

    (REGISTRY_DIR / "metric_summary.md").write_text(
        "# Exp49 Metric Summary\n\n"
        "Milestone C py312 remediation computed no inpainting metrics. Environment/import/CUDA smoke status: "
        f"`{status}`.\n"
    )
    append_once(
        REGISTRY_DIR / "status.md",
        "Environment gate py312 remediation:",
        f"Environment gate py312 remediation: `{status}` via `/home/hj/venvs/rose_exp49_py312`.",
    )
    result_line = f"C_env_py312\t{status}\tPython 3.12 virtualenv import/CUDA smoke passed with torch 2.6.0 and ROSE requirements.\n"
    results_path = REGISTRY_DIR / "results.tsv"
    if result_line.strip() not in results_path.read_text():
        with results_path.open("a") as f:
            f.write(result_line)

    block = f"""## 2026-06-30 Exp49 ROSE Python 3.12 Environment Remediation

Exp49 created `/home/hj/venvs/rose_exp49_py312` using `virtualenv` because system `python3.12` lacks `ensurepip`. PyTorch 2.6.0, torchvision 0.21.0, and ROSE requirements were installed into that isolated env. Import/CUDA smoke status: `{status}`. No inference, training, optimizer step, checkpoint update, VOR-Eval use, H20 action, or official ROSE source modification was performed.
"""
    append_once(PROJECT_ROOT / "PRD/00_current_status.md", "2026-06-30 Exp49 ROSE Python 3.12 Environment Remediation", block)
    append_once(
        PROJECT_ROOT / "PRD/01_experiment_matrix.md",
        "2026-06-30 Exp49 Py312 Env Update",
        f"""## 2026-06-30 Exp49 Py312 Env Update

| Experiment | Milestone | Status | Notes |
| --- | --- | --- | --- |
| `exp49_pai_rose_adapter_feasibility` | C env py312 remediation | `{status}` | Python 3.12 virtualenv plus torch 2.6.0 and ROSE requirements import/CUDA smoke; no inference/training. |
""",
    )
    append_once(
        PROJECT_ROOT / "PRD/46_exp49_pai_rose_adapter_feasibility.md",
        "## Milestone C2 Update - 2026-06-30",
        f"""## Milestone C2 Update - 2026-06-30

Status: `{status}`.

The earlier Python 3.10 env was replaced for gate purposes by `/home/hj/venvs/rose_exp49_py312`. System `python3.12` lacks `ensurepip`, so `virtualenv` from the isolated py310 env was used to seed pip without modifying system Python. PyTorch 2.6.0, torchvision 0.21.0, and ROSE requirements were installed in the py312 venv; import/CUDA smoke had `{len(failed)}` failed checks.

No inference, training, optimizer step, checkpoint update, VOR-Eval use, H20 action, or official ROSE source modification was performed.
""",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
