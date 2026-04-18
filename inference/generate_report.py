# -*- coding: utf-8 -*-
"""
generate_report.py — Experiment results report generator

Usage:
    python generate_report.py dir1 dir2 ... dirN

Reads summary.json from each experiment directory,
outputs experiment_report.md with OR/BR tables grouped by weight type.
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime


PIXEL_KEYS = ["psnr_mean", "ssim_mean", "lpips_mean", "ewarp", "as_mean", "is_mean"]
PIXEL_LABELS = {"psnr_mean": "PSNR↑", "ssim_mean": "SSIM↑", "lpips_mean": "LPIPS↓",
                "ewarp": "Ewarp↓", "as_mean": "AS↑", "is_mean": "IS↑"}
VBENCH_DIMS = [
    "subject_consistency", "background_consistency",
    "temporal_flickering", "motion_smoothness",
    "aesthetic_quality", "imaging_quality",
]
VBENCH_LABELS = {
    "subject_consistency": "Subj_Con↑", "background_consistency": "BG_Con↑",
    "temporal_flickering": "Temp_Flk↑", "motion_smoothness": "Mot_Smo↑",
    "aesthetic_quality": "Aesth_Q↑", "imaging_quality": "Img_Q↑",
}

# Weight prefix patterns (order matters: longest prefix first)
WEIGHT_PREFIXES = ["FT_S2_48K", "FT_S2_34K", "FT_S2_26K", "FT_S2_8K", "Finetune", "Orign"]


def _is_dual_mode(per_video):
    """Check if per_video entries use baseline_/text_ prefixed keys."""
    if not per_video:
        return False
    sample = per_video[0]
    return any(k.startswith(("baseline_", "text_")) for k in sample)


def _get_vbench(entry, mode_prefix=None):
    if mode_prefix:
        d = entry.get(f"{mode_prefix}_vbench")
        if d:
            return d
    return entry.get("vbench", {})


def _get_metrics(entry, mode_prefix=None):
    if mode_prefix:
        d = entry.get(f"{mode_prefix}_metrics")
        if d:
            return d
    return entry.get("metrics", {})


def load_summary(exp_dir):
    p = Path(exp_dir) / "summary.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def avg_metric(per_video, mode_prefix, key):
    vals = []
    for v in per_video:
        val = _get_metrics(v, mode_prefix).get(key)
        if val is None:
            val = _get_vbench(v, mode_prefix).get(key)
        if val is not None and isinstance(val, (int, float)) and val >= 0:
            vals.append(val)
    return sum(vals) / len(vals) if vals else None


def fmt(v, prec=4):
    if v is None:
        return "N/A"
    return f"{v:.{prec}f}"


def parse_exp_name(name):
    """Parse experiment dir name to extract metadata.

    Supports naming: {Weight}_{steps}_{Dataset}_{blend}_{dil}_{gs}
    Also supports the older smallcfg_*/normalcfg_* experiment names.
    """
    info = {"name": name, "weight": "", "cfg_type": "", "steps": "",
            "dataset": "", "blend": "", "dilation": "", "gs": ""}

    # 解析权重前缀
    name_rest = name
    for prefix in WEIGHT_PREFIXES:
        if name.startswith(prefix + "_"):
            info["weight"] = prefix
            name_rest = name[len(prefix) + 1:]
            break

    if not info["weight"]:
        info["weight"] = "Default" if name.startswith(("smallcfg_", "normalcfg_")) else "Unknown"

    parts = name_rest.split("_")

    # 解析 steps/cfg: s2=2-Step, s4=4-Step, n4=Normal CFG 4-Step
    if name_rest.startswith("s2_"):
        info["steps"] = "2-Step"
        info["cfg_type"] = "PCM"
    elif name_rest.startswith("s4_"):
        info["steps"] = "4-Step"
        info["cfg_type"] = "PCM"
    elif name_rest.startswith("n4_"):
        info["steps"] = "NormalCFG-4Step"
        info["cfg_type"] = "NormalCFG"
    elif name_rest.startswith("smallcfg"):
        info["cfg_type"] = "smallcfg"
    elif name_rest.startswith("normalcfg"):
        info["cfg_type"] = "normalcfg"

    # 解析 dataset
    for p in parts:
        if p in ("OR", "BR"):
            info["dataset"] = p
        elif p == "2step":
            info["steps"] = "2-Step"
        elif p == "4step":
            info["steps"] = "4-Step"

    # 解析 blend/dilation
    if "noblend" in name_rest:
        info["blend"] = "No"
        info["dilation"] = "0"
    elif "blend" in name_rest:
        info["blend"] = "Yes"
    if "dil8" in name_rest:
        info["dilation"] = "8"
    if "nodil" in name_rest:
        info["dilation"] = "0"

    # 解析 gs
    for p in parts:
        if p.startswith("gs"):
            info["gs"] = p[2:]

    return info


def _vbench_avg_val(per_video, mode_prefix):
    """Compute average VBench score across all dimensions."""
    vs = [avg_metric(per_video, mode_prefix, d) for d in VBENCH_DIMS]
    vs = [x for x in vs if x is not None]
    return sum(vs) / len(vs) if vs else 0


def generate_detailed_table(experiments, dataset_filter, has_gt):
    """Generate markdown table for one dataset type (OR or BR)."""
    filtered = [(name, data) for name, data in experiments
                if parse_exp_name(name)["dataset"] == dataset_filter]
    if not filtered:
        return ""

    lines = []

    dual_mode = any(_is_dual_mode(data.get("per_video", [])) for _, data in filtered)
    if dual_mode:
        modes = [("baseline", "BL"), ("text", "TG")]
    else:
        modes = [(None, None)]

    # Sort by VBench average descending
    filtered.sort(key=lambda nd: _vbench_avg_val(nd[1].get("per_video", []), modes[0][0]),
                  reverse=True)

    # VBench table
    lines.append(f"### VBench Scores ({dataset_filter})")
    lines.append("")
    header = "| Weight | Config |"
    if dual_mode:
        header += " Mode |"
    for dim in VBENCH_DIMS:
        header += f" {VBENCH_LABELS[dim]} |"
    header += " **Avg** |"
    lines.append(header)

    n_cols = len(VBENCH_DIMS) + (4 if dual_mode else 3)
    sep = "|" + "|".join(["---"] * n_cols) + "|"
    lines.append(sep)

    for name, data in filtered:
        info = parse_exp_name(name)
        per_video = data.get("per_video", [])
        config_short = f"{info['steps']}_{'blend' if info['blend']=='Yes' else 'noblend'}_gs{info['gs']}"

        for mode_prefix, mode_label in modes:
            row = f"| **{info['weight']}** | {config_short} |"
            if dual_mode:
                row += f" {mode_label} |"
            vals = []
            for dim in VBENCH_DIMS:
                v = avg_metric(per_video, mode_prefix, dim)
                row += f" {fmt(v)} |"
                if v is not None:
                    vals.append(v)
            avg_val = sum(vals) / len(vals) if vals else None
            row += f" **{fmt(avg_val)}** |"
            lines.append(row)

    lines.append("")

    # Pixel metrics table (BR only)
    if has_gt:
        lines.append(f"### Pixel Metrics ({dataset_filter}, GT available)")
        lines.append("")
        header = "| Weight | Config |"
        if dual_mode:
            header += " Mode |"
        for pk in PIXEL_KEYS:
            header += f" {PIXEL_LABELS[pk]} |"
        lines.append(header)

        n_cols = len(PIXEL_KEYS) + (3 if dual_mode else 2)
        sep = "|" + "|".join(["---"] * n_cols) + "|"
        lines.append(sep)

        px_sorted = sorted(filtered,
                           key=lambda nd: avg_metric(nd[1].get("per_video", []), modes[0][0], "psnr_mean") or 0,
                           reverse=True)
        for name, data in px_sorted:
            info = parse_exp_name(name)
            per_video = data.get("per_video", [])
            config_short = f"{info['steps']}_{'blend' if info['blend']=='Yes' else 'noblend'}_gs{info['gs']}"

            for mode_prefix, mode_label in modes:
                row = f"| **{info['weight']}** | {config_short} |"
                if dual_mode:
                    row += f" {mode_label} |"
                for pk in PIXEL_KEYS:
                    v = avg_metric(per_video, mode_prefix, pk)
                    row += f" {fmt(v)} |"
                lines.append(row)

        lines.append("")

    return "\n".join(lines)


def generate_cross_exp_comparison(experiments):
    """Generate cross-experiment comparison summary, sorted by VBench average."""
    lines = []
    dual_mode = any(_is_dual_mode(data.get("per_video", [])) for _, data in experiments)

    lines.append("## Cross-Experiment Comparison (VBench Average)")
    lines.append("")
    if dual_mode:
        lines.append("Average VBench score across all videos, **Text-Guided (TG)** mode only:")
    else:
        lines.append("Average VBench score across all videos:")
    lines.append("")

    header = "| Weight | Dataset | Blend | Dil | Steps | GS |"
    for dim in VBENCH_DIMS:
        header += f" {VBENCH_LABELS[dim]} |"
    header += " **Avg** |"
    lines.append(header)

    n_cols = len(VBENCH_DIMS) + 7
    sep = "|" + "|".join(["---"] * n_cols) + "|"
    lines.append(sep)

    mode_prefix = "text" if dual_mode else None

    sorted_exps = sorted(experiments,
                         key=lambda nd: _vbench_avg_val(nd[1].get("per_video", []), mode_prefix),
                         reverse=True)

    for name, data in sorted_exps:
        info = parse_exp_name(name)
        per_video = data.get("per_video", [])

        row = (f"| **{info['weight']}** | {info['dataset']} | {info['blend']} | "
               f"{info['dilation']} | {info['steps']} | {info['gs']} |")
        vals = []
        for dim in VBENCH_DIMS:
            v = avg_metric(per_video, mode_prefix, dim)
            row += f" {fmt(v)} |"
            if v is not None:
                vals.append(v)
        avg_val = sum(vals) / len(vals) if vals else None
        row += f" **{fmt(avg_val)}** |"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def generate_weight_comparison(experiments):
    """Generate weight-vs-weight comparison for the same config."""
    lines = []
    lines.append("## Weight Comparison (Same Config)")
    lines.append("")

    # 动态发现所有权重类型
    all_weights = sorted(set(parse_exp_name(name)["weight"] for name, _ in experiments))
    if not all_weights:
        return ""

    lines.append(f"Compare {' vs '.join(all_weights)} on the same config/dataset:")
    lines.append("")

    # Group experiments by (steps, dataset, blend, dilation, gs)
    config_groups = {}
    for name, data in experiments:
        info = parse_exp_name(name)
        key = (info["steps"], info["dataset"], info["blend"], info["dilation"], info["gs"])
        if key not in config_groups:
            config_groups[key] = {}
        config_groups[key][info["weight"]] = (name, data)

    if not config_groups:
        return ""

    dual_mode = any(_is_dual_mode(data.get("per_video", [])) for _, data in experiments)
    mode_prefix = "text" if dual_mode else None

    header = "| Config | Dataset |"
    for w in all_weights:
        header += f" {w} Avg |"
    lines.append(header)
    sep = "|---|---|"
    for _ in all_weights:
        sep += "---|"
    lines.append(sep)

    for (steps, dataset, blend, dil, gs), weight_dict in sorted(config_groups.items()):
        blend_str = "blend" if blend == "Yes" else "noblend"
        config_str = f"{steps}_{blend_str}_gs{gs}"
        row = f"| {config_str} | {dataset} |"
        for w in all_weights:
            if w in weight_dict:
                _, d = weight_dict[w]
                pv = d.get("per_video", [])
                avg = _vbench_avg_val(pv, mode_prefix)
                row += f" **{fmt(avg)}** |"
            else:
                row += " N/A |"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_report.py exp_dir1 exp_dir2 ...")
        sys.exit(1)

    exp_dirs = sys.argv[1:]
    experiments = []

    for d in exp_dirs:
        data = load_summary(d)
        if data is None:
            print(f"[WARN] summary.json not found in {d}, skipping.")
            continue
        name = os.path.basename(d.rstrip("/"))
        experiments.append((name, data))

    if not experiments:
        print("[ERROR] No valid experiment data found.")
        sys.exit(1)

    print(f"Loaded {len(experiments)} experiment(s).")

    # Count weight types
    weight_counts = {}
    for name, _ in experiments:
        w = parse_exp_name(name)["weight"]
        weight_counts[w] = weight_counts.get(w, 0) + 1
    weight_summary = ", ".join(f"{k}: {v}" for k, v in sorted(weight_counts.items()))

    report = []
    report.append(f"# Experiment Comparison Report")
    report.append(f"")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"")
    report.append(f"**{len(experiments)} experiments** — {weight_summary}")
    report.append(f"")
    report.append(f"## Experiment Configuration")
    report.append(f"")
    report.append(f"| # | Weight | Directory | ckpt | GS | Dataset | Blend | Dilation | Videos |")
    report.append(f"|---|--------|-----------|------|----|---------|-------|----------|--------|")

    for i, (name, data) in enumerate(experiments, 1):
        cfg = data.get("config", {})
        info = parse_exp_name(name)
        n = data.get("num_videos", 0)
        report.append(f"| {i} | **{info['weight']}** | `{name}` | {cfg.get('ckpt', '?')} | "
                       f"{cfg.get('text_guidance_scale', '?')} | {info['dataset']} | "
                       f"{info['blend']} | {info['dilation']} | {n} |")

    report.append("")

    # OR section
    or_exps = [(n, d) for n, d in experiments if parse_exp_name(n)["dataset"] == "OR"]
    if or_exps:
        report.append("---")
        report.append("")
        report.append("## Object Removal (OR) Results")
        report.append("")
        report.append(generate_detailed_table(experiments, "OR", has_gt=False))

    # BR section
    br_exps = [(n, d) for n, d in experiments if parse_exp_name(n)["dataset"] == "BR"]
    if br_exps:
        report.append("---")
        report.append("")
        report.append("## Background Restoration (BR) Results")
        report.append("")
        report.append(generate_detailed_table(experiments, "BR", has_gt=True))

    # Cross-experiment comparison
    report.append("---")
    report.append("")
    report.append(generate_cross_exp_comparison(experiments))

    # Weight comparison table
    report.append("---")
    report.append("")
    report.append(generate_weight_comparison(experiments))

    # Summary
    report.append("---")
    report.append("")
    report.append("## Experiment Summary")
    report.append("")
    report.append("### Key Findings")
    report.append("")
    report.append("| Comparison | Question |")
    report.append("|-----------|----------|")
    report.append("| Orign vs FT_S2_26K | Does Stage2 26K-step finetuning improve quality? |")
    report.append("| Orign vs FT_S2_8K | Does Stage2 8K-step finetuning improve quality? |")
    report.append("| Orign vs FT_S2_34K | Does Stage2 34K-step finetuning improve quality? |")
    report.append("| Orign vs FT_S2_48K | Does Stage2 48K-step finetuning improve quality? |")
    report.append("| FT_S2_8K vs FT_S2_26K vs FT_S2_34K vs FT_S2_48K | Which training step count is optimal? |")
    report.append("| 2-Step vs 4-Step | Does increasing inference steps improve quality? |")
    report.append("| noblend vs blend+dil8 | Does mask blending improve visual quality? |")
    report.append("")
    report.append("> Review the tables above to answer these questions based on your metric priorities.")
    report.append("")

    out_path = "experiment_report.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"Report saved to: {out_path}")
    print(f"Total experiments: {len(experiments)}")


if __name__ == "__main__":
    main()
