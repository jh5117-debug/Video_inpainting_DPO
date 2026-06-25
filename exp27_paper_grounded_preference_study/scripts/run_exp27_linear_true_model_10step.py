#!/usr/bin/env python3
"""Run true-model Linear-DPO Frozen/EMA 1/10-step gates for Exp27.

This is a micro gate, not a long study. It reuses the true DiffuEraser
Stage1 policy/reference forward path from the SDPO parity runner and applies
the official Linear-DPO utility to one fixed real preference batch.
"""

from __future__ import annotations

import argparse
import csv
import copy
import gc
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import torch
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.dpo.train_stage1 import collate_fn, import_model_class_from_model_name_or_path  # noqa: E402
from exp27_paper_grounded_preference_study.code.official_parity import (  # noqa: E402
    linear_dpo_clip_ratio,
    linear_dpo_loss,
)
from exp27_paper_grounded_preference_study.scripts.run_exp27_true_model_objective_parity import (  # noqa: E402
    Paths,
    clear_models,
    dtype_from_name,
    encode_batch,
    forward_pair,
    load_state,
    make_dataset,
    mse_losses,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--report-dir", type=Path, default=Path("reports"))
    p.add_argument("--base-model", default="/mnt/nas/hj/weights/stable-diffusion-v1-5")
    p.add_argument("--vae", default="/mnt/nas/hj/weights/sd-vae-ft-mse")
    p.add_argument("--sft-weights", default="/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000")
    p.add_argument(
        "--manifest",
        default="/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/"
        "exp08c_youtubevos_gtwin_d3comp_lose_fixed_pai/manifests/selected_primary_comp.gtwin.pai_paths.jsonl",
    )
    p.add_argument("--row-index", type=int, default=0)
    p.add_argument("--timestep", type=int, default=500)
    p.add_argument("--nframes", type=int, default=16)
    p.add_argument("--height", type=int, default=320)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seed", type=int, default=20260625)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-7)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--beta-dpo", type=float, default=5000.0)
    p.add_argument("--eta-dpo", type=float, default=0.01)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    return p.parse_args()


def trainable_parameters(policy: tuple) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for module in policy:
        params.extend([p for p in module.parameters() if p.requires_grad])
    return params


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += float(p.grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(total)


def state_vector_norm(policy: tuple, initial: list[dict[str, torch.Tensor]]) -> float:
    total = 0.0
    for module, before in zip(policy, initial):
        for key, value in module.state_dict().items():
            b = before[key].to(device=value.device, dtype=value.dtype)
            total += float((value.detach().float() - b.float()).pow(2).sum().cpu())
    return math.sqrt(total)


def clone_state(policy: tuple) -> list[dict[str, torch.Tensor]]:
    return [{k: v.detach().cpu().clone() for k, v in module.state_dict().items()} for module in policy]


def ema_update_reference(ref: tuple, policy: tuple, decay: float) -> float:
    max_delta = 0.0
    with torch.no_grad():
        for ref_module, policy_module in zip(ref, policy):
            for ref_param, policy_param in zip(ref_module.parameters(), policy_module.parameters()):
                before = ref_param.detach().float().clone()
                ref_param.data.mul_(decay).add_(policy_param.detach().data.to(dtype=ref_param.dtype), alpha=1.0 - decay)
                max_delta = max(max_delta, float((ref_param.detach().float() - before).abs().max().cpu()))
    return max_delta


def compute_linear_loss(policy: tuple, ref: tuple, tensors: dict, dtype: torch.dtype, beta: float, eta: float):
    model_pred, ref_pred = forward_pair(policy, ref, tensors, dtype)
    noise = tensors["noise"]
    m_w, m_l = mse_losses(model_pred, noise)
    r_w, r_l = mse_losses(ref_pred, noise)
    loss = linear_dpo_loss(m_w, m_l, r_w, r_l, beta_dpo=beta, eta_dpo=eta)
    utility = linear_dpo_clip_ratio(m_w.detach() - m_l.detach(), r_w.detach() - r_l.detach(), beta_dpo=beta, eta_dpo=eta)
    return loss, {
        "loss": float(loss.detach().float().cpu()),
        "utility_mean": float(utility.float().mean().cpu()),
        "utility_min": float(utility.float().min().cpu()),
        "utility_max": float(utility.float().max().cpu()),
        "winner_policy_mse": float(m_w.mean().detach().float().cpu()),
        "loser_policy_mse": float(m_l.mean().detach().float().cpu()),
        "winner_ref_mse": float(r_w.mean().detach().float().cpu()),
        "loser_ref_mse": float(r_l.mean().detach().float().cpu()),
        "margin": float(((m_l - m_w) - (r_l - r_w)).mean().detach().float().cpu()),
        "finite": bool(torch.isfinite(model_pred).all().item() and torch.isfinite(ref_pred).all().item() and torch.isfinite(loss).all().item()),
    }


def run_variant(args, variant: str, dataset, vae, noise_scheduler, text_encoder, paths: Paths, device, dtype) -> tuple[list[dict], dict]:
    policy, ref, identity = load_state("S0", paths, device, dtype)
    initial_policy = clone_state(policy)
    initial_ref = clone_state(ref)
    params = trainable_parameters(policy)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sample = dataset[args.row_index]
    batch = collate_fn([sample])
    tensors = encode_batch(batch, vae, noise_scheduler, text_encoder, args.nframes, args.timestep, dtype, args.seed + 17, device)
    rows: list[dict] = []
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        loss, diag = compute_linear_loss(policy, ref, tensors, dtype, args.beta_dpo, args.eta_dpo)
        if not diag["finite"]:
            raise FloatingPointError(f"non-finite Linear-DPO loss at step {step}")
        loss.backward()
        gnorm = grad_norm(params)
        torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        opt.step()
        ema_delta = 0.0
        if variant == "linear_ema":
            ema_delta = ema_update_reference(ref, policy, args.ema_decay)
        row = {
            "variant": variant,
            "step": step,
            "sample_id": sample.get("sample_id"),
            "row_index": args.row_index,
            "timestep": args.timestep,
            "lr": args.lr,
            "beta_dpo": args.beta_dpo,
            "eta_dpo": args.eta_dpo,
            "ema_decay": args.ema_decay if variant == "linear_ema" else "",
            "grad_norm": gnorm,
            "ema_step_max_delta": ema_delta,
            **diag,
            "policy_delta_norm": state_vector_norm(policy, initial_policy),
            "reference_delta_norm": state_vector_norm(ref, initial_ref),
        }
        rows.append(row)
    summary = {
        "variant": variant,
        "status": "passed" if rows and all(r["finite"] for r in rows) and rows[-1]["policy_delta_norm"] > 0 else "failed",
        "records": len(rows),
        "step1_loss": rows[0]["loss"] if rows else None,
        "step10_loss": rows[-1]["loss"] if rows else None,
        "step10_policy_delta_norm": rows[-1]["policy_delta_norm"] if rows else None,
        "step10_reference_delta_norm": rows[-1]["reference_delta_norm"] if rows else None,
        "max_grad_norm": max((r["grad_norm"] for r in rows), default=None),
        "identity": identity,
    }
    if variant == "linear_frozen" and summary["step10_reference_delta_norm"] != 0.0:
        summary["status"] = "failed_reference_changed"
    if variant == "linear_ema" and (summary["step10_reference_delta_norm"] or 0.0) <= 0.0:
        summary["status"] = "failed_ema_no_reference_update"
    del tensors
    clear_models(policy, ref)
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for true-model Linear-DPO")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    dtype = dtype_from_name(args.dtype)
    torch.manual_seed(args.seed)
    paths = Paths(args.base_model, args.vae, args.sft_weights, args.sft_weights, args.manifest)
    text_cls = import_model_class_from_model_name_or_path(args.base_model, None)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, subfolder="tokenizer", use_fast=False)
    text_encoder = text_cls.from_pretrained(args.base_model, subfolder="text_encoder").to(device=device, dtype=dtype).eval().requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(args.vae, subfolder="vae").to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")
    dataset = make_dataset(args, tokenizer)
    all_rows: list[dict] = []
    summaries: list[dict] = []
    for variant in ("linear_frozen", "linear_ema"):
        rows, summary = run_variant(args, variant, dataset, vae, noise_scheduler, text_encoder, paths, device, dtype)
        all_rows.extend(rows)
        summaries.append(summary)
        gc.collect()
        torch.cuda.empty_cache()
    write_csv(args.output_dir / "linear_true_model_10step.csv", all_rows)
    summary = {
        "status": "LINEAR_TRUE_MODEL_1_10_STEP_PASSED" if all(s["status"] == "passed" for s in summaries) else "FAILED",
        "summaries": summaries,
        "rows": len(all_rows),
        "note": "True DiffuEraser Stage1 policy/reference forward, one fixed real batch, no RC-FPO and no long training.",
    }
    (args.output_dir / "linear_true_model_10step_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.report_dir / "exp27_linear_frozen_10step.csv", [r for r in all_rows if r["variant"] == "linear_frozen"])
    write_csv(args.report_dir / "exp27_linear_ema_10step.csv", [r for r in all_rows if r["variant"] == "linear_ema"])
    (args.report_dir / "exp27_linear_true_model_10step.md").write_text(
        "# Exp27 Linear-DPO True Model 1/10-Step Gate\n\n"
        f"- status: `{summary['status']}`\n"
        f"- output_dir: `{args.output_dir}`\n"
        f"- variants: `linear_frozen`, `linear_ema`\n"
        f"- steps: `{args.steps}`\n"
        f"- row_index: `{args.row_index}`\n"
        f"- timestep: `{args.timestep}`\n"
        f"- lr: `{args.lr}`\n"
        f"- beta_dpo: `{args.beta_dpo}`\n"
        f"- eta_dpo: `{args.eta_dpo}`\n"
        f"- ema_decay: `{args.ema_decay}`\n\n"
        "This is a technical true-model micro gate. It does not start RC-FPO or any 50-step/long study.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"].endswith("PASSED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
