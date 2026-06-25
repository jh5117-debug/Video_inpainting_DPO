#!/usr/bin/env python3
"""Exp27 LocalDPO DiffuEraser 24F adaptation lane.

This runner is intentionally staged:

1. Generate P8/P32 LocalDPO 24F pairs with official moving masks.
2. Require P32 gate before objective training.
3. Run only original LocalDPO-style 1-step and 10-step micro objectives.

It does not run 50-step studies, O0-O5, four-grid runs, or RC-FPO.
"""

from __future__ import annotations

import argparse
import copy
import csv
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

from training.dpo.train_stage1 import (  # noqa: E402
    build_region_loss_weight_map,
    collate_fn,
    compute_dpo_loss,
    import_model_class_from_model_name_or_path,
)
from exp27_paper_grounded_preference_study.code.localdpo_24f_adaptation import (  # noqa: E402
    P8_GATE,
    P32_GATE,
    classify_pair,
    composite_outside_clean,
    controlled_corruption_preview,
    load_frames,
    make_condition_frames,
    official_localdpo_masks,
    read_jsonl,
    run_diffueraser_loser,
    safe_id,
    save_frames,
    save_review_assets,
    select_manifest_rows,
    sha256_tree,
    summarize_gate,
    write_csv,
    write_jsonl,
)
from exp27_paper_grounded_preference_study.scripts.run_exp27_true_model_objective_parity import (  # noqa: E402
    Paths,
    clear_models,
    dtype_from_name,
    encode_batch,
    forward_pair,
    load_state,
    make_dataset,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["p8", "p32", "objective", "all"], default="all")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--report-dir", type=Path, default=Path("reports"))
    p.add_argument(
        "--source-manifest",
        type=Path,
        default=Path(
            "/mnt/nas/hj/H20_Video_inpainting_DPO/data/generated_losers/"
            "exp08c_youtubevos_gtwin_d3comp_lose_fixed_pai/manifests/selected_primary_comp.gtwin.pai_paths.jsonl"
        ),
    )
    p.add_argument("--seed", type=int, default=20260625)
    p.add_argument("--frames", type=int, default=24)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--width", type=int, default=432)
    p.add_argument("--run-diffueraser-loser", action="store_true")
    p.add_argument("--allow-controlled-preview-loser", action="store_true")
    p.add_argument("--allow-official-mask-fallback", action="store_true")
    p.add_argument("--allow-objective-with-pending-gate", action="store_true")
    p.add_argument("--project-root", type=Path, default=REPO_ROOT)
    p.add_argument("--base-model", type=Path, default=Path("/mnt/nas/hj/weights/stable-diffusion-v1-5"))
    p.add_argument("--vae", type=Path, default=Path("/mnt/nas/hj/weights/sd-vae-ft-mse"))
    p.add_argument("--sft-weights", type=Path, default=Path("/mnt/nas/hj/weights/diffuEraser/converted_weights_step48000"))
    p.add_argument("--propainter-model-dir", type=Path, default=Path("/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter"))
    p.add_argument("--pcm-weights-path", type=Path, default=Path("/mnt/nas/hj/data/third_party_video_inpainting/weights/propainter/raft-things.pth"))
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--row-index", type=int, default=0)
    p.add_argument("--timestep", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-7)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--beta-dpo", type=float, default=500.0)
    p.add_argument("--global-dpo-weight", type=float, default=1.0)
    p.add_argument("--ra-dpo-weight", type=float, default=1.0)
    p.add_argument("--sft-reg-weight", type=float, default=0.1)
    p.add_argument("--mask-region-weight", type=float, default=1.0)
    p.add_argument("--boundary-region-weight", type=float, default=0.5)
    p.add_argument("--outside-region-weight", type=float, default=0.05)
    return p.parse_args()


def gate_for_phase(phase: str):
    return P8_GATE if phase == "p8" else P32_GATE


def write_markdown_report(path: Path, title: str, summary: dict, manifest_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# {title}\n\n"
        f"- status: `{summary['status']}`\n"
        f"- pairs: `{summary['pairs']}`\n"
        f"- technical_valid: `{summary['technical_valid']}`\n"
        f"- medium_hard: `{summary['medium_hard']}`\n"
        f"- hard_plausible: `{summary['hard_plausible']}`\n"
        f"- trivial_bad: `{summary['trivial_bad']}`\n"
        f"- technical_invalid: `{summary['technical_invalid']}`\n"
        f"- global_collapse: `{summary['global_collapse']}`\n"
        f"- outside_preservation_passed: `{summary['outside_preservation_passed']}`\n"
        f"- video_review: `{summary['video_review']}`\n"
        f"- manifest: `{manifest_path}`\n\n"
        "This is LocalDPO DiffuEraser 24F adaptation evidence. It is not RC-FPO, "
        "not O0-O5, and not a 50-step study.\n",
        encoding="utf-8",
    )


def generate_gate_pairs(args: argparse.Namespace, gate_name: str) -> tuple[dict, Path]:
    gate = gate_for_phase(gate_name)
    rows = read_jsonl(args.source_manifest)
    selected = select_manifest_rows(rows, gate.required_pairs, args.seed + gate.required_pairs)
    root = args.output_dir / "pairs" / gate.name.lower()
    pair_rows = []
    for ordinal, (source_index, row) in enumerate(selected):
        sample_id = safe_id(row.get("sample_id") or row.get("video_id"), f"row{source_index:06d}")
        pair_id = f"localdpo24f_{source_index:06d}_{sample_id}"
        pair_root = root / pair_id
        clean_dir = pair_root / "clean_winner_24f"
        mask_dir = pair_root / "localdpo_corruption_mask"
        condition_dir = pair_root / "masked_condition"
        raw_loser_dir = pair_root / "self_model_loser_raw"
        loser_dir = pair_root / "self_model_loser_outside_reinjected"
        review_dir = pair_root / "review"
        clean = load_frames(Path(str(row["win_video_path"])), args.frames, (args.width, args.height), mode="RGB")
        masks, mask_meta = official_localdpo_masks(
            seed=args.seed + source_index,
            frames=args.frames,
            height=args.height,
            width=args.width,
            require_official=not args.allow_official_mask_fallback,
        )
        condition = make_condition_frames(clean, masks)
        save_frames(clean, clean_dir)
        save_frames(masks, mask_dir)
        save_frames(condition, condition_dir)
        loser_status = "pending_self_model_loser"
        error = ""
        if args.run_diffueraser_loser:
            try:
                input_video_root = pair_root / "diffueraser_input" / "videos"
                input_mask_root = pair_root / "diffueraser_input" / "masks"
                save_frames(clean, input_video_root / pair_id)
                save_frames(masks, input_mask_root / pair_id)
                run_diffueraser_loser(
                    project_root=args.project_root,
                    video_root=input_video_root,
                    mask_root=input_mask_root,
                    output_dir=raw_loser_dir,
                    work_dir=pair_root / "diffueraser_work",
                    prompt=str(row.get("prompt") or ""),
                    frames=args.frames,
                    height=args.height,
                    width=args.width,
                    base_model=args.base_model,
                    vae=args.vae,
                    diffueraser_weights=args.sft_weights,
                    propainter_model_dir=args.propainter_model_dir,
                    pcm_weights_path=args.pcm_weights_path,
                )
                raw_loser = load_frames(raw_loser_dir, args.frames, (args.width, args.height), mode="RGB")
                loser = composite_outside_clean(raw_loser, clean, masks)
                save_frames(loser, loser_dir)
                loser_status = "self_model_loser_generated"
            except Exception as exc:  # noqa: BLE001 - pair-level gate records the failure.
                error = repr(exc)
        elif args.allow_controlled_preview_loser:
            loser = controlled_corruption_preview(clean, masks, args.seed + source_index)
            save_frames(loser, loser_dir)
            loser_status = "controlled_preview_loser_not_gate_valid"
        if loser_status.endswith("generated") or loser_status.startswith("controlled_preview"):
            loser_loaded = load_frames(loser_dir, args.frames, (args.width, args.height), mode="RGB")
            metrics = classify_pair(clean, masks, loser_loaded)
            if loser_status.startswith("controlled_preview"):
                metrics["technical_valid"] = False
                metrics["classification"] = "TECHNICAL_INVALID"
                metrics["outside_preservation_passed"] = False
            review_assets = save_review_assets(pair_id=pair_id, clean=clean, masks=masks, loser=loser_loaded, review_dir=review_dir)
        else:
            metrics = {
                "technical_valid": False,
                "classification": "TECHNICAL_INVALID",
                "mask_psnr": "",
                "outside_psnr": "",
                "global_psnr": "",
                "mask_area": "",
                "outside_preservation_passed": False,
                "global_collapse": False,
            }
            review_assets = {}
        out_row = {
            "sample_id": pair_id,
            "source_sample_id": row.get("sample_id"),
            "source_row_index": source_index,
            "pair_index": ordinal,
            "prompt": row.get("prompt") or "",
            "win_video_path": str(clean_dir),
            "final_loser_video_path": str(loser_dir),
            "mask_path": str(mask_dir),
            "condition_video_path": str(condition_dir),
            "raw_self_model_loser_path": str(raw_loser_dir),
            "generation_source": "diffueraser_only",
            "generation_model": "diffueraser",
            "localdpo_frames": args.frames,
            "localdpo_mask_semantics": "official 3D Bezier/random-shape moving corruption mask",
            "outside_reinjection": "rgb outside clean composite after DiffuEraser self-model loser generation",
            "winner_identity": "real clean winner",
            "loser_identity": loser_status,
            "mask_sha256": sha256_tree(mask_dir),
            "clean_sha256": sha256_tree(clean_dir),
            "loser_sha256": sha256_tree(loser_dir) if loser_dir.exists() and list(loser_dir.glob("*.png")) else "",
            "mask_meta": mask_meta,
            "error": error,
            "review_assets": review_assets,
            **metrics,
        }
        pair_rows.append(out_row)
    manifest_path = root / f"exp27_localdpo_24f_{gate.name.lower()}_manifest.jsonl"
    write_jsonl(manifest_path, pair_rows)
    write_csv(root / f"exp27_localdpo_24f_{gate.name.lower()}_pairs.csv", pair_rows)
    summary = summarize_gate(gate, pair_rows)
    summary.update(
        {
            "manifest": str(manifest_path),
            "source_manifest": str(args.source_manifest),
            "run_diffueraser_loser": bool(args.run_diffueraser_loser),
            "controlled_preview_allowed": bool(args.allow_controlled_preview_loser),
        }
    )
    (root / f"exp27_localdpo_24f_{gate.name.lower()}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown_report(root / f"exp27_localdpo_24f_{gate.name.lower()}_gate.md", f"Exp27 LocalDPO 24F {gate.name} Gate", summary, manifest_path)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.report_dir / f"exp27_localdpo_24f_{gate.name.lower()}_pairs.csv", pair_rows)
    write_markdown_report(
        args.report_dir / f"exp27_localdpo_24f_{gate.name.lower()}_gate.md",
        f"Exp27 LocalDPO 24F {gate.name} Gate",
        summary,
        manifest_path,
    )
    return summary, manifest_path


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


def clone_state(policy: tuple) -> list[dict[str, torch.Tensor]]:
    return [{k: v.detach().cpu().clone() for k, v in module.state_dict().items()} for module in policy]


def state_vector_norm(policy: tuple, initial: list[dict[str, torch.Tensor]]) -> float:
    total = 0.0
    for module, before in zip(policy, initial):
        for key, value in module.state_dict().items():
            base = before[key].to(device=value.device, dtype=value.dtype)
            total += float((value.detach().float() - base.float()).pow(2).sum().cpu())
    return math.sqrt(total)


def localdpo_objective_loss(policy: tuple, ref: tuple, tensors: dict, batch: dict, dtype: torch.dtype, args: argparse.Namespace):
    model_pred, ref_pred = forward_pair(policy, ref, tensors, dtype)
    noise = tensors["noise"]
    global_loss, global_diag = compute_dpo_loss(
        model_pred,
        ref_pred,
        noise,
        beta_dpo=args.beta_dpo,
        sft_reg_weight=0.0,
        nframes=args.frames,
    )
    weight_map, region_stats = build_region_loss_weight_map(
        batch["masks"],
        mask_region_weight=args.mask_region_weight,
        boundary_region_weight=args.boundary_region_weight,
        outside_region_weight=args.outside_region_weight,
    )
    ra_loss, ra_diag = compute_dpo_loss(
        model_pred,
        ref_pred,
        noise,
        loss_weight_map=weight_map,
        loss_region_mode="weighted",
        region_stats=region_stats,
        beta_dpo=args.beta_dpo,
        sft_reg_weight=args.sft_reg_weight,
        nframes=args.frames,
    )
    loss = float(args.global_dpo_weight) * global_loss + float(args.ra_dpo_weight) * ra_loss
    diag = {
        "total_loss": float(loss.detach().float().cpu()),
        "global_dpo_loss": float(global_diag["dpo_loss"]),
        "ra_dpo_loss": float(ra_diag["dpo_loss"]),
        "sft_loss": float(ra_diag["sft_loss"]),
        "implicit_acc": float(global_diag["implicit_acc"]),
        "ra_implicit_acc": float(ra_diag["implicit_acc"]),
        "mask_area_ratio": float(region_stats["mask_area_ratio"]),
        "boundary_area_ratio": float(region_stats["boundary_area_ratio"]),
        "outside_area_ratio": float(region_stats["outside_area_ratio"]),
        "mask_region_mse": float(region_stats.get("mask_region_mse", 0.0)),
        "boundary_region_mse": float(region_stats.get("boundary_region_mse", 0.0)),
        "outside_region_mse": float(region_stats.get("outside_region_mse", 0.0)),
    }
    return loss, diag


def run_objective_variant(args: argparse.Namespace, manifest: Path, steps: int, dataset, vae, scheduler, text_encoder, paths, device, dtype):
    policy, ref, identity = load_state("S0", paths, device, dtype)
    initial_policy = clone_state(policy)
    initial_ref = clone_state(ref)
    params = trainable_parameters(policy)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    rows = []
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        sample = dataset[(args.row_index + step - 1) % len(dataset)]
        batch = collate_fn([sample])
        tensors = encode_batch(batch, vae, scheduler, text_encoder, args.frames, args.timestep, dtype, args.seed + step, device)
        loss, diag = localdpo_objective_loss(policy, ref, tensors, batch, dtype, args)
        if not torch.isfinite(loss).all().item():
            raise FloatingPointError(f"non-finite LocalDPO objective at step {step}")
        loss.backward()
        gnorm = grad_norm(params)
        torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        opt.step()
        rows.append(
            {
                "variant": f"localdpo_original_{steps}step",
                "step": step,
                "sample_id": sample.get("sample_id"),
                "manifest": str(manifest),
                "timestep": args.timestep,
                "lr": args.lr,
                "beta_dpo": args.beta_dpo,
                "global_dpo_weight": args.global_dpo_weight,
                "ra_dpo_weight": args.ra_dpo_weight,
                "sft_reg_weight": args.sft_reg_weight,
                "grad_norm": gnorm,
                "policy_delta_norm": state_vector_norm(policy, initial_policy),
                "reference_delta_norm": state_vector_norm(ref, initial_ref),
                **diag,
            }
        )
        del tensors
        gc.collect()
        torch.cuda.empty_cache()
    summary = {
        "variant": f"localdpo_original_{steps}step",
        "status": "passed" if rows and all(math.isfinite(float(r["total_loss"])) for r in rows) and rows[-1]["policy_delta_norm"] > 0 else "failed",
        "records": len(rows),
        "step_last_loss": rows[-1]["total_loss"] if rows else None,
        "step_last_policy_delta_norm": rows[-1]["policy_delta_norm"] if rows else None,
        "step_last_reference_delta_norm": rows[-1]["reference_delta_norm"] if rows else None,
        "max_grad_norm": max((r["grad_norm"] for r in rows), default=None),
        "identity": identity,
    }
    if summary["step_last_reference_delta_norm"] != 0.0:
        summary["status"] = "failed_reference_changed"
    clear_models(policy, ref)
    del initial_policy, initial_ref
    gc.collect()
    torch.cuda.empty_cache()
    return rows, summary


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_objective(args: argparse.Namespace, manifest: Path) -> dict:
    summary_path = args.output_dir / "pairs" / "p32" / "exp27_localdpo_24f_p32_summary.json"
    if summary_path.exists():
        p32_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if p32_summary.get("status") != "P32_PASSED" and not args.allow_objective_with_pending_gate:
            raise RuntimeError(f"P32 gate has not passed: {summary_path}")
    elif not args.allow_objective_with_pending_gate:
        raise FileNotFoundError(f"P32 summary is required before objective: {summary_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LocalDPO true-model objective")
    device = torch.device("cuda")
    dtype = dtype_from_name(args.dtype)
    torch.manual_seed(args.seed)
    paths = Paths(str(args.base_model), str(args.vae), str(args.sft_weights), str(args.sft_weights), str(manifest))
    text_cls = import_model_class_from_model_name_or_path(str(args.base_model), None)
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), subfolder="tokenizer", use_fast=False)
    text_encoder = text_cls.from_pretrained(str(args.base_model), subfolder="text_encoder").to(device=device, dtype=dtype).eval().requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(str(args.vae), subfolder="vae").to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    scheduler = DDPMScheduler.from_pretrained(str(args.base_model), subfolder="scheduler")
    objective_args = copy.copy(args)
    objective_args.preference_manifest = str(manifest)
    objective_args.nframes = args.frames
    objective_args.train_height = args.height
    objective_args.train_width = args.width
    objective_args.resolution = args.width
    objective_args.train_mask_mode = "partial"
    objective_args.mask_from_manifest = True
    objective_args.videodpo_full_mask_value = 0.0
    objective_args.max_resample_attempts = 128
    objective_args.proportion_empty_prompts = 0.0
    dataset = make_dataset(objective_args, tokenizer)
    all_rows: list[dict] = []
    summaries: list[dict] = []
    for steps in (1, 10):
        rows, summary = run_objective_variant(args, manifest, steps, dataset, vae, scheduler, text_encoder, paths, device, dtype)
        all_rows.extend(rows)
        summaries.append(summary)
    out_dir = args.output_dir / "objective"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows_csv(out_dir / "localdpo_original_1_10_step.csv", all_rows)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_rows_csv(args.report_dir / "exp27_localdpo_24f_original_1_10_step.csv", all_rows)
    status = "LOCALDPO_24F_ORIGINAL_OBJECTIVE_1_10_STEP_PASSED" if all(s["status"] == "passed" for s in summaries) else "LOCALDPO_24F_ORIGINAL_OBJECTIVE_FAILED"
    summary = {
        "status": status,
        "manifest": str(manifest),
        "summaries": summaries,
        "rows": len(all_rows),
        "note": "Original LocalDPO-style RA-DPO + global DPO + SFT micro objective only; no RC-FPO and no 50-step study.",
    }
    (out_dir / "localdpo_original_1_10_step_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.report_dir / "exp27_localdpo_24f_original_1_10_step.md").write_text(
        "# Exp27 LocalDPO 24F Original Objective 1/10-Step\n\n"
        f"- status: `{status}`\n"
        f"- manifest: `{manifest}`\n"
        f"- output_dir: `{out_dir}`\n"
        "- objective: `RA-DPO + global DPO + SFT`\n"
        "- steps: `1`, `10`\n"
        "- RC-FPO: `NOT_STARTED`\n"
        "- 50-step: `NOT_STARTED`\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    last_manifest = args.output_dir / "pairs" / "p32" / "exp27_localdpo_24f_p32_manifest.jsonl"
    results = []
    if args.phase in {"p8", "all"}:
        summary, _ = generate_gate_pairs(args, "p8")
        results.append(summary)
        if args.phase == "p8":
            print(json.dumps({"status": summary["status"], "results": results}, indent=2, sort_keys=True))
            return 0 if summary["status"] == "P8_PASSED" else 2
        if summary["status"] != "P8_PASSED":
            print(json.dumps({"status": "P8_BLOCKED_P32", "results": results}, indent=2, sort_keys=True))
            return 2
    if args.phase in {"p32", "all"}:
        summary, last_manifest = generate_gate_pairs(args, "p32")
        results.append(summary)
        if args.phase == "p32":
            print(json.dumps({"status": summary["status"], "results": results}, indent=2, sort_keys=True))
            return 0 if summary["status"] == "P32_PASSED" else 2
        if summary["status"] != "P32_PASSED":
            print(json.dumps({"status": "P32_BLOCKED_OBJECTIVE", "results": results}, indent=2, sort_keys=True))
            return 2
    if args.phase in {"objective", "all"}:
        objective_summary = run_objective(args, last_manifest)
        results.append(objective_summary)
    final_status = results[-1]["status"] if results else "NOOP"
    print(json.dumps({"status": final_status, "results": results}, indent=2, sort_keys=True))
    return 0 if str(final_status).endswith("PASSED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
