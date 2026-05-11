#!/usr/bin/env python3
from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import pandas as pd


ROOT = Path("/home/hj")
OUT_DIR = ROOT / "Video_inpainting_DPO/PRD/assets/dpo_metric_analysis_20260505"
DOC_PATH = ROOT / "Video_inpainting_DPO/PRD/dpo_metric_regularization_prd_20260505.md"
GLOBAL_STEP_CUTOFF = 10000

LOGS = {
    "diffdpo": {
        "label": "DiffDPO_loss",
        "path": ROOT / "log/普通DiffDPO_loss.log",
        "kind": "diffueraser",
        "analysis_cutoff_step": GLOBAL_STEP_CUTOFF,
        "analysis_note": "global-step aligned to step<=10000.",
    },
    "no_lose_gap": {
        "label": "DiffDPO_no_lose_gap",
        "path": ROOT / "log/把lose_gap删除的loss.log",
        "kind": "diffueraser",
        "analysis_cutoff_step": GLOBAL_STEP_CUTOFF,
        "analysis_note": "global-step aligned to step<=10000.",
    },
    "videodpo": {
        "label": "VideoDPO_open_source",
        "path": ROOT / "log/VideoDPO的训练.log",
        "kind": "videodpo",
        "analysis_cutoff_step": GLOBAL_STEP_CUTOFF,
        "analysis_note": "old log records optimizer global_step; use it directly as the x-axis.",
    },
    "videodpo_inpaint_data": {
        "label": "VideoDPO_on_VideoInpainting_data",
        "path": ROOT / "log/使用VideoInpainting的数据集的VideoDPO的loss.log",
        "kind": "videodpo",
        "analysis_cutoff_step": GLOBAL_STEP_CUTOFF,
        "analysis_note": (
            "old log records optimizer global_step; use it directly as the x-axis. "
            "New mixed logs combine [dpo_diag] rows and Lightning progress rows, both aligned to optimizer global_step."
        ),
    },
}

DISPLAY_NAMES = {
    "diffdpo": "DiffDPO",
    "no_lose_gap": "去掉 lose_gap 实验",
    "videodpo": "开源 VideoDPO",
    "videodpo_inpaint_data": "VideoDPO 使用 Inpainting 数据",
}

METRIC_DISPLAY_NAMES = {
    "implicit_acc": "implicit_acc",
    "l_dpo": "DPO_loss",
    "win_gap": "winner gap",
    "lose_gap": "loser gap",
    "mse_w": "winner MSE",
    "ref_mse_w": "ref winner MSE",
    "mse_l": "loser MSE",
    "ref_mse_l": "ref loser MSE",
    "loser_dominant_ratio": "loser dominant ratio",
}

METRIC_ALIASES = {
    "L_total": "l_total",
    "L_dpo": "l_dpo",
    "L_sft": "l_sft",
    "Implicit Acc": "implicit_acc",
    "Win Gap": "win_gap",
    "Lose Gap": "lose_gap",
    "Reward Margin": "reward_margin",
    "Sigma Term": "sigma_term",
    "KL Divergence": "kl_divergence",
    "MSE (win)": "mse_w",
    "MSE (lose)": "mse_l",
    "Ref MSE (win)": "ref_mse_w",
    "Ref MSE (lose)": "ref_mse_l",
    "DGR (grad norm)": "dgr_grad_norm",
    "InsideTerm mean": "inside_term_mean",
    "InsideTerm min": "inside_term_min",
    "InsideTerm max": "inside_term_max",
    "Loser Dominant %": "loser_dominant_ratio",
}

VIDEODPO_KEY_ALIASES = {
    "mw": "mse_w",
    "ml": "mse_l",
    "mwref": "ref_mse_w",
    "mrefw": "ref_mse_w",
    "mlref": "ref_mse_l",
    "mrefl": "ref_mse_l",
    "inside_term": "inside_term_mean",
}


def _float(x: str) -> float:
    return float(x.replace(",", ""))


def parse_diffueraser(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    lines = path.read_text(errors="replace").splitlines()
    rows: list[dict[str, float | int | str]] = []
    val_rows: list[dict[str, float | int | str]] = []
    current: dict[str, float | int | str] | None = None

    metric_re = re.compile(
        r"\[(?:R0|G)\]\s+(.+?)\s+([-+]?(?:\d+\.\d+|\d+)(?:[eE][-+]?\d+)?)"
    )
    step_re = re.compile(r"DPO Diagnostics @ Step\s+(\d+)")
    samples_re = re.compile(r"Samples:\s*(\d+)/(\d+)\s+correct,\s*(\d+)\s+loser-dominant")
    val_re = re.compile(r"Validation @ Step\s+(\d+)")
    avg_re = re.compile(r"Avg\s*\|\s*([-+]?\d+\.\d+)\s*\|\s*([-+]?\d+\.\d+)")

    i = 0
    while i < len(lines):
        line = lines[i]
        sm = step_re.search(line)
        if sm:
            if current:
                rows.append(current)
            current = {"step": int(sm.group(1))}
            i += 1
            continue

        if current is not None:
            mm = metric_re.search(line)
            if mm:
                name = mm.group(1).strip()
                key = METRIC_ALIASES.get(name)
                if key:
                    current[key] = _float(mm.group(2))
            smp = samples_re.search(line)
            if smp:
                current["correct"] = int(smp.group(1))
                current["samples"] = int(smp.group(2))
                current["loser_dominant_count"] = int(smp.group(3))
            if "════" in line and "DPO Diagnostics" not in line:
                # Keep scanning because borders appear at the top too; the next step header
                # is the reliable flush point.
                pass

        vm = val_re.search(line)
        if vm:
            step = int(vm.group(1))
            for j in range(i + 1, min(i + 12, len(lines))):
                am = avg_re.search(lines[j])
                if am:
                    val_rows.append(
                        {"step": step, "psnr": _float(am.group(1)), "ssim": _float(am.group(2))}
                    )
                    break
        i += 1

    if current:
        rows.append(current)

    diag = pd.DataFrame(rows).drop_duplicates(subset=["step"], keep="last").sort_values("step")
    vals = pd.DataFrame(val_rows).drop_duplicates(subset=["step"], keep="last").sort_values("step")
    return diag.reset_index(drop=True), vals.reset_index(drop=True)


def _axis_step_from_videodpo_line(line: str, fallback_step: int) -> int:
    if gm := re.search(r"global_step=(\d+)", line):
        return int(gm.group(1))
    return fallback_step


def parse_videodpo(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    val_rows: list[dict[str, float | int | str]] = []
    val_fail_rows: list[dict[str, float | int | str]] = []
    number = r"[-+]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))(?:[eE][-+]?\d+)?"
    keyval_re = re.compile(rf"([A-Za-z0-9_]+)=({number}(?:/\d+)?)")
    progress_re = re.compile(r"Epoch\s+(\d+):.*?\|\s*(\d+)/(\d+)\s+\[")
    progress_keyval_re = re.compile(rf"([A-Za-z0-9_/]+)=({number})")
    progress_aliases = {
        "global/implicit_acc": "implicit_acc",
        "rank0/dpo_loss": "dpo_loss",
        "rank0/win_gap": "win_gap",
        "rank0/lose_gap": "lose_gap",
        "train/loss_simple_step": "l_dpo",
        "train/loss_step": "l_total",
    }
    val_re = re.compile(
        rf"\[video_inpaint_val\]\s+step=(\d+).*?psnr=({number}).*?ssim=({number})"
    )
    fail_re = re.compile(r"\[video_inpaint_val\]\s+step=(\d+).*?failed:\s+(.+)")
    for line in path.read_text(errors="replace").splitlines():
        if "[video_inpaint_val]" in line:
            if vm := val_re.search(line):
                val_rows.append(
                    {
                        "step": _axis_step_from_videodpo_line(line, int(vm.group(1))),
                        "psnr": _float(vm.group(2)),
                        "ssim": _float(vm.group(3)),
                    }
                )
            elif fm := fail_re.search(line):
                val_fail_rows.append(
                    {
                        "step": _axis_step_from_videodpo_line(line, int(fm.group(1))),
                        "error": fm.group(2).strip(),
                    }
                )

        if pm := progress_re.search(line):
            epoch = int(pm.group(1))
            batch_idx = int(pm.group(2))
            steps_per_epoch = int(pm.group(3))
            # Lightning may print a batch_idx=0 progress row at an epoch boundary
            # with stale metrics from the previous epoch. Keep real diagnostics for
            # step 0 and avoid treating those reset rows as new measurements.
            if batch_idx == 0:
                continue
            row: dict[str, float | int | str] = {
                "step": epoch * steps_per_epoch + batch_idx,
                "epoch": epoch,
                "epoch_batch_idx": batch_idx,
                "steps_per_epoch": steps_per_epoch,
                "step_source": "lightning_progress",
            }
            for key, val in progress_keyval_re.findall(line):
                canonical_key = progress_aliases.get(key)
                if canonical_key:
                    row[canonical_key] = _float(val)
            if "dpo_loss" in row and "l_dpo" not in row:
                row["l_dpo"] = row["dpo_loss"]
            if "l_dpo" in row:
                rows.append(row)

        if "[dpo_diag]" in line:
            row: dict[str, float | int | str] = {}
            for key, val in keyval_re.findall(line):
                canonical_key = VIDEODPO_KEY_ALIASES.get(key, key)
                if key == "step":
                    row["pair_step"] = int(val.split("/")[0])
                    if "step" not in row:
                        row["step"] = int(val.split("/")[0])
                        row["step_source"] = "step_fallback"
                elif key == "global_step":
                    global_step = int(val.split("/")[0])
                    row["global_step"] = global_step
                    row["step"] = global_step
                    row["step_source"] = "global_step"
                elif key == "implicit_acc_count" and "/" in val:
                    correct, total = val.split("/", 1)
                    row["implicit_acc_correct"] = int(correct)
                    row["implicit_acc_total"] = int(total)
                    row["implicit_acc_count"] = int(correct) / max(int(total), 1)
                elif "/" in val:
                    row[canonical_key] = _float(val.split("/", 1)[0])
                else:
                    row[canonical_key] = _float(val)
            if "step" in row:
                rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["step"], keep="last").sort_values("step")
    if "dpo_loss" in df.columns:
        df["l_dpo"] = df["dpo_loss"]
    if "total_loss" in df.columns:
        df["l_total"] = df["total_loss"]
    vals = (
        pd.DataFrame(val_rows).drop_duplicates(subset=["step"], keep="last").sort_values("step")
        if val_rows
        else pd.DataFrame(columns=["step", "psnr", "ssim"])
    )
    val_fails = (
        pd.DataFrame(val_fail_rows).drop_duplicates(subset=["step"], keep="last").sort_values("step")
        if val_fail_rows
        else pd.DataFrame(columns=["step", "error"])
    )
    return df.reset_index(drop=True), vals.reset_index(drop=True), val_fails.reset_index(drop=True)


def phase_summary(df: pd.DataFrame, label: str) -> pd.DataFrame:
    n = len(df)
    take = max(1, math.ceil(n * 0.2))
    parts = [("first_20pct", df.head(take)), ("last_20pct", df.tail(take))]
    cols = [
        "l_dpo",
        "implicit_acc",
        "win_gap",
        "lose_gap",
        "mse_w",
        "ref_mse_w",
        "mse_l",
        "ref_mse_l",
        "loser_dominant_ratio",
    ]
    rows = []
    for name, part in parts:
        row = {
            "experiment": label,
            "phase": name,
            "step_range": f"{int(part.step.min())}-{int(part.step.max())}",
            "n": len(part),
        }
        for col in cols:
            if col in part:
                row[col] = part[col].mean()
        if "win_gap" in part and "lose_gap" in part:
            row["lose_minus_win_gap"] = (part["lose_gap"] - part["win_gap"]).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def delta_summary(df: pd.DataFrame, vals: pd.DataFrame | None, label: str) -> dict[str, float | str | int]:
    first = df.iloc[0]
    last = df.iloc[-1]
    row: dict[str, float | str | int] = {
        "experiment": label,
        "first_step": int(first["step"]),
        "last_step": int(last["step"]),
    }
    for col in ["l_dpo", "implicit_acc", "win_gap", "lose_gap", "mse_w", "mse_l", "ref_mse_w", "ref_mse_l"]:
        if col in df.columns:
            valid = df[["step", col]].dropna()
            if valid.empty:
                continue
            first_valid = valid.iloc[0]
            last_valid = valid.iloc[-1]
            row[f"{col}_first"] = first_valid[col]
            row[f"{col}_last"] = last_valid[col]
            row[f"{col}_delta"] = last_valid[col] - first_valid[col]
            row[f"{col}_first_step"] = int(first_valid["step"])
            row[f"{col}_last_step"] = int(last_valid["step"])
    if vals is not None and not vals.empty:
        row["psnr_first"] = vals.iloc[0]["psnr"]
        row["psnr_last"] = vals.iloc[-1]["psnr"]
        row["psnr_best"] = vals["psnr"].max()
        row["ssim_first"] = vals.iloc[0]["ssim"]
        row["ssim_last"] = vals.iloc[-1]["ssim"]
        row["ssim_best"] = vals["ssim"].max()
    return row


def nearest_diag_for_vals(diag: pd.DataFrame, vals: pd.DataFrame, label: str) -> pd.DataFrame:
    if vals.empty:
        return pd.DataFrame()
    rows = []
    for _, v in vals.iterrows():
        idx = (diag["step"] - v["step"]).abs().idxmin()
        d = diag.loc[idx]
        row = {"experiment": label, "val_step": int(v["step"]), "nearest_diag_step": int(d["step"])}
        for col in ["l_dpo", "implicit_acc", "win_gap", "lose_gap", "mse_w", "mse_l", "ref_mse_w", "ref_mse_l", "loser_dominant_ratio"]:
            if col in d:
                row[col] = d[col]
        row["psnr"] = v["psnr"]
        row["ssim"] = v["ssim"]
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_by_step(df: pd.DataFrame, chunk: int = 300) -> pd.DataFrame:
    part = df.copy()
    part["step_bin"] = (part["step"] // chunk) * chunk
    numeric_cols = [c for c in part.columns if c != "step_bin" and pd.api.types.is_numeric_dtype(part[c])]
    out = part.groupby("step_bin", as_index=False)[numeric_cols].mean()
    out["step"] = out["step_bin"]
    return out.drop(columns=["step_bin"])


def chunk_summary(df: pd.DataFrame, label: str, chunk: int = 300) -> pd.DataFrame:
    part = df.copy()
    part["chunk"] = (part["step"] // chunk) * chunk
    cols = ["l_dpo", "implicit_acc", "win_gap", "lose_gap", "mse_w", "ref_mse_w", "mse_l", "ref_mse_l", "loser_dominant_ratio"]
    agg = {col: "mean" for col in cols if col in part.columns}
    out = part.groupby("chunk", as_index=False).agg(agg)
    ranges = part.groupby("chunk")["step"].agg(["min", "max"]).reset_index()
    out = out.merge(ranges, on="chunk", how="left")
    out["experiment"] = label
    out["step_range"] = out.apply(lambda r: f"{int(r['min'])}-{int(r['max'])}", axis=1)
    return out[["experiment", "step_range"] + [c for c in cols if c in out.columns]]


def fmt_df(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    show = df.copy()
    for c in show.columns:
        if pd.api.types.is_float_dtype(show[c]):
            show[c] = show[c].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    show = show.fillna("").astype(str)
    headers = list(show.columns)
    rows = show.values.tolist()
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(cell.replace("\n", " ") for cell in row) + " |")
    return "\n".join(out)


def plot_all(diags: dict[str, pd.DataFrame], vals: dict[str, pd.DataFrame]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    cjk_font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if cjk_font_path.exists():
        fm.fontManager.addfont(str(cjk_font_path))
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

    def plot_metric(ax, df: pd.DataFrame, metric: str, label: str, *, linestyle: str = "-", color=None) -> pd.Series:
        x = df["step"]
        y = df[metric]
        if len(df) > 600:
            window = max(25, len(df) // 120)
            smooth_y = y.rolling(window, min_periods=1, center=True).mean()
            line = ax.plot(
                x,
                smooth_y,
                label=f"{label} ({window}-step mean)",
                linewidth=1.9,
                linestyle=linestyle,
                color=color,
            )[0]
            ax.plot(
                x,
                y,
                linewidth=0.35,
                alpha=0.12,
                linestyle=linestyle,
                color=line.get_color(),
            )
            return smooth_y
        else:
            ax.plot(x, y, label=label, linewidth=1.6, linestyle=linestyle, color=color)
            return y

    def set_robust_ylim(ax, series_list: list[pd.Series], *, include_zero: bool = False, pad_ratio: float = 0.12) -> None:
        values = pd.concat([s.dropna() for s in series_list if s is not None and len(s.dropna())], ignore_index=True)
        if values.empty:
            return
        lo = float(values.quantile(0.01))
        hi = float(values.quantile(0.99))
        if include_zero:
            lo = min(lo, 0.0)
            hi = max(hi, 0.0)
        if math.isclose(lo, hi):
            delta = abs(hi) * 0.1 + 1e-6
            lo -= delta
            hi += delta
        pad = (hi - lo) * pad_ratio
        ax.set_ylim(lo - pad, hi + pad)

    experiments = list(diags.keys())
    fig, axes = plt.subplots(6, len(experiments), figsize=(6.3 * len(experiments), 22), constrained_layout=True)
    if len(experiments) == 1:
        axes = axes.reshape(6, 1)
    titles = {
        "diffdpo": "DiffDPO loss",
        "no_lose_gap": "No lose-gap ablation",
        "videodpo": "VideoDPO",
        "videodpo_inpaint_data": "VideoDPO + inpainting data",
    }
    row_defs = [
        ("implicit_acc", ["implicit_acc"], "Implicit acc"),
        ("loss", ["l_dpo"], "DPO loss"),
        ("gaps", ["win_gap", "lose_gap"], "Gap vs ref"),
        ("winner_mse", ["mse_w", "ref_mse_w"], "Winner MSE"),
        ("loser_mse", ["mse_l", "ref_mse_l"], "Loser MSE"),
        ("quality", ["psnr", "ssim"], "Validation quality"),
    ]
    for col, key in enumerate(experiments):
        df = aggregate_by_step(diags[key], 300) if key.startswith("videodpo") and len(diags[key]) > 20 else diags[key]
        robust_scale = key.startswith("videodpo")
        for row, (_, metrics, ylabel) in enumerate(row_defs):
            ax = axes[row, col]
            if row == 0:
                ax.set_title(titles[key], fontsize=13, fontweight="bold")
            if metrics == ["psnr", "ssim"]:
                v = vals.get(key, pd.DataFrame())
                if v is not None and not v.empty:
                    ax.plot(v["step"], v["psnr"], marker="o", label="PSNR")
                    ax2 = ax.twinx()
                    ax2.plot(v["step"], v["ssim"], marker="s", color="#dd8452", label="SSIM")
                    ax.set_ylabel("PSNR")
                    ax2.set_ylabel("SSIM")
                    ax.set_xlim(0, GLOBAL_STEP_CUTOFF)
                    ax2.set_xlim(0, GLOBAL_STEP_CUTOFF)
                    lines, labels = ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)
                else:
                    ax.text(0.5, 0.5, "No PSNR/SSIM in log", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                continue
            plotted_series = []
            for metric in metrics:
                if metric in df.columns:
                    plotted_series.append(plot_metric(ax, df, metric, METRIC_DISPLAY_NAMES.get(metric, metric)))
            if metrics == ["win_gap", "lose_gap"]:
                ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
                if robust_scale:
                    set_robust_ylim(ax, plotted_series, include_zero=True)
            if metrics == ["l_dpo"] and robust_scale:
                set_robust_ylim(ax, plotted_series, include_zero=False)
            if metrics in (["mse_w", "ref_mse_w"], ["mse_l", "ref_mse_l"]) and robust_scale:
                set_robust_ylim(ax, plotted_series, include_zero=False)
            if metrics == ["implicit_acc"]:
                ax.axhline(0.5, color="black", linewidth=0.8, alpha=0.5)
                if robust_scale:
                    set_robust_ylim(ax, plotted_series, include_zero=False)
                else:
                    ax.set_ylim(-0.02, 1.05)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("global-step")
            ax.set_xlim(0, GLOBAL_STEP_CUTOFF)
            ax.legend(loc="best", fontsize=8)
    fig.suptitle("DPO diagnostics across four real training logs", fontsize=16, fontweight="bold")
    fig.savefig(OUT_DIR / "all_experiments_metric_panels.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for key, raw_df in diags.items():
        df = aggregate_by_step(raw_df, 300) if key.startswith("videodpo") and len(raw_df) > 20 else raw_df
        label = DISPLAY_NAMES[key]
        smooth = max(1, len(df) // 120)
        x = df["step"]
        axes[0, 0].plot(x, df["implicit_acc"].rolling(smooth, min_periods=1).mean(), label=label)
        axes[0, 1].plot(x, df["l_dpo"].rolling(smooth, min_periods=1).mean(), label=label)
        axes[1, 0].plot(x, df["win_gap"].rolling(smooth, min_periods=1).mean(), label=f"{label} 的 winner gap")
        axes[1, 0].plot(x, df["lose_gap"].rolling(smooth, min_periods=1).mean(), linestyle="--", label=f"{label} 的 loser gap")
        if "loser_dominant_ratio" in df:
            axes[1, 1].plot(x, df["loser_dominant_ratio"].rolling(smooth, min_periods=1).mean(), label=label)
    axes[0, 0].axhline(0.5, color="black", linewidth=0.8, alpha=0.5)
    axes[1, 0].axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axes[0, 0].set_title("Implicit acc")
    axes[0, 1].set_title("DPO loss")
    axes[1, 0].set_title("Win/Lose gap")
    axes[1, 1].set_title("Loser dominant ratio")
    for ax in axes.ravel():
        ax.set_xlabel("global-step")
        ax.set_xlim(0, GLOBAL_STEP_CUTOFF)
        ax.legend(fontsize=8)
    fig.savefig(OUT_DIR / "comparison_overlay_smoothed.png", dpi=180)
    plt.close(fig)

    video_rows = [
        ("implicit_acc", ["implicit_acc"], "Implicit acc"),
        ("loss", ["l_dpo"], "DPO loss"),
        ("gaps", ["win_gap", "lose_gap"], "Gap vs ref"),
        ("winner_mse", ["mse_w", "ref_mse_w"], "Winner MSE"),
        ("loser_mse", ["mse_l", "ref_mse_l"], "Loser MSE"),
        ("loser_dominant", ["loser_dominant_ratio"], "Loser dominant ratio"),
    ]
    for video_key, filename, title in [
        ("videodpo", "videodpo_300step_metric_panels.png", "开源 VideoDPO metrics (300-step mean)"),
        (
            "videodpo_inpaint_data",
            "videodpo_inpainting_data_300step_metric_panels.png",
            "VideoDPO 使用 Inpainting 数据 metrics (main comparable window)",
        ),
    ]:
        if video_key not in diags:
            continue
        video_df = aggregate_by_step(diags[video_key], 300) if len(diags[video_key]) > 20 else diags[video_key]
        fig, axes = plt.subplots(6, 1, figsize=(11, 16), constrained_layout=True)
        for ax, (_, metrics, ylabel) in zip(axes, video_rows):
            plotted = []
            for metric in metrics:
                if metric in video_df.columns:
                    ax.plot(
                        video_df["step"],
                        video_df[metric],
                        marker="o",
                        markersize=3.2,
                        linewidth=1.7,
                        label=METRIC_DISPLAY_NAMES.get(metric, metric),
                    )
                    plotted.append(video_df[metric])
            if metrics == ["win_gap", "lose_gap"]:
                ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
                set_robust_ylim(ax, plotted, include_zero=True)
            elif metrics == ["implicit_acc"]:
                ax.axhline(0.5, color="black", linewidth=0.8, alpha=0.5)
                set_robust_ylim(ax, plotted, include_zero=False)
            else:
                set_robust_ylim(ax, plotted, include_zero=False)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("global-step")
            ax.set_xlim(0, GLOBAL_STEP_CUTOFF)
            ax.legend(loc="best", fontsize=8)
        fig.suptitle(title, fontsize=15, fontweight="bold")
        fig.savefig(OUT_DIR / filename, dpi=180)
        plt.close(fig)


def build_doc(
    diags: dict[str, pd.DataFrame],
    vals: dict[str, pd.DataFrame],
    phase_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    chunks_df: pd.DataFrame,
    scope_df: pd.DataFrame,
    validation_failures_df: pd.DataFrame,
) -> str:
    table_files = {
        "phase_summary": OUT_DIR / "phase_summary.csv",
        "delta_summary": OUT_DIR / "delta_summary.csv",
        "validation_aligned": OUT_DIR / "validation_aligned.csv",
        "videodpo_300step_chunks": OUT_DIR / "videodpo_300step_chunks.csv",
        "all_diagnostics": OUT_DIR / "all_diagnostics.csv",
        "all_diagnostics_full": OUT_DIR / "all_diagnostics_full.csv",
        "experiment_scope": OUT_DIR / "experiment_scope.csv",
        "validation_failures": OUT_DIR / "validation_failures.csv",
    }
    diffdpo_last = phase_df[(phase_df["experiment"] == "DiffDPO_loss") & (phase_df["phase"] == "last_20pct")].iloc[0]
    no_lose_last = phase_df[(phase_df["experiment"] == "DiffDPO_no_lose_gap") & (phase_df["phase"] == "last_20pct")].iloc[0]
    video_last = phase_df[(phase_df["experiment"] == "VideoDPO_open_source") & (phase_df["phase"] == "last_20pct")].iloc[0]
    video_inpaint_last = phase_df[
        (phase_df["experiment"] == "VideoDPO_on_VideoInpainting_data") & (phase_df["phase"] == "last_20pct")
    ].iloc[0]
    diffdpo_delta = delta_df[delta_df["experiment"] == "DiffDPO_loss"].iloc[0]
    no_lose_delta = delta_df[delta_df["experiment"] == "DiffDPO_no_lose_gap"].iloc[0]
    video_delta = delta_df[delta_df["experiment"] == "VideoDPO_open_source"].iloc[0]
    video_inpaint_delta = delta_df[delta_df["experiment"] == "VideoDPO_on_VideoInpainting_data"].iloc[0]
    failure_text = (
        fmt_df(validation_failures_df)
        if not validation_failures_df.empty
        else "当前四个日志里没有记录 validation failure。"
    )

    return f"""# Video Inpainting DPO Diagnostics and Regularized-DPO PRD

Date: 2026-05-07

2026-05-09 update:

- 最新项目交接入口见 `PRD/README_FOR_NEXT_CHAT.md`、`PRD/NEXT_CHAT_FULL_CONTEXT_20260509.md` 和 `PRD/PROJECT_HANDOFF_20260509.md`。
- HAL 本地代码已经把 DiffDPO stage1/stage2 的 `implicit_acc` 诊断改成 video-pair 粒度。
- 当前改动只影响 diagnostics，不改变 DPO loss 本身的 frame-level 优化目标。
- 本文中的历史实验图仍来自旧日志，其中 DiffDPO 旧日志的 `implicit_acc` 是 frame-level 统计；新实验需要用更新后的 pair-level 口径重新画图。

Source logs:

- `/home/hj/log/普通DiffDPO_loss.log`
- `/home/hj/log/把lose_gap删除的loss.log`
- `/home/hj/log/VideoDPO的训练.log`
- `/home/hj/log/使用VideoInpainting的数据集的VideoDPO的loss.log`

Generated artifacts:

- `assets/dpo_metric_analysis_20260505/all_experiments_metric_panels.png`
- `assets/dpo_metric_analysis_20260505/comparison_overlay_smoothed.png`
- `assets/dpo_metric_analysis_20260505/videodpo_300step_metric_panels.png`
- `assets/dpo_metric_analysis_20260505/videodpo_inpainting_data_300step_metric_panels.png`
- CSV tables in `assets/dpo_metric_analysis_20260505/`

![All metric panels](assets/dpo_metric_analysis_20260505/all_experiments_metric_panels.png)

![Smoothed comparison](assets/dpo_metric_analysis_20260505/comparison_overlay_smoothed.png)

![VideoDPO 300-step panels](assets/dpo_metric_analysis_20260505/videodpo_300step_metric_panels.png)

![VideoDPO on VideoInpainting data 300-step panels](assets/dpo_metric_analysis_20260505/videodpo_inpainting_data_300step_metric_panels.png)

## 1. Executive Summary

这四个真实训练日志现在给出的结论更清楚了：

1. **普通 DiffDPO_loss 出现了典型的 loser-dominant failure。** 训练目标很快被满足，`implicit_acc` 进入接近 1 的区域，`dpo_loss` 贴近 0；但 `win_gap` 和 `lose_gap` 同时变大，且 `lose_gap` 远大于 `win_gap`。这说明模型主要靠把 loser 变得更差来赢。在当前 `global-step<=10000` 对齐窗口内，PSNR/SSIM 从最早 validation 的 `{diffdpo_delta['psnr_first']:.4f} / {diffdpo_delta['ssim_first']:.4f}` 掉到最后的 `{diffdpo_delta['psnr_last']:.4f} / {diffdpo_delta['ssim_last']:.4f}`。

2. **删除 lose_gap 的 ablation 证明 winner 分支本身不是坏的。** 最后 20% 诊断里 `win_gap={no_lose_last['win_gap']:.6f}`，仍然小于 0；PSNR/SSIM 保持在 `24.19 / 0.866` 左右。这说明当不允许模型靠 loser 侧获利时，winner 侧可以稳定工作。

3. **开源 VideoDPO 也不是“winner MSE 必须优于 ref”的形态。** 它最后 20% 的 `win_gap={video_last['win_gap']:.6f}`、`lose_gap={video_last['lose_gap']:.6f}`，二者都为正，但尺度只有 1e-3 量级。VideoDPO 的偏好优化成功不是因为每一步 denoising MSE 都比 ref 好，而是最终 sampled video 的偏好指标更好。对我们这个 video inpainting 任务，不能直接把 VideoDPO 的现象当作安全证明，因为我们的普通 DiffDPO 已经在 PSNR/SSIM 上实际崩了。

4. **VideoDPO 换成 VideoInpainting 数据后，也按 global-step 直接看，不再按 epoch 裁剪。** 这里的横坐标统一使用训练框架里的 optimizer `global_step`。最新日志里既有每 300 个 global-step 的 `[dpo_diag]`，也有 Lightning 进度条里的逐步 `rank0/dpo_loss / win_gap / lose_gap`；本报告把二者合并到同一个 global-step 轴上。当前已完成日志最后 20% 的 `win_gap={video_inpaint_last['win_gap']:.6f}`、`lose_gap={video_inpaint_last['lose_gap']:.6f}`，`loser_dominant_ratio={video_inpaint_last['loser_dominant_ratio']:.6f}`，说明统一数据集后 VideoDPO 仍然会出现强烈的 loser shortcut 倾向。

因此下一步不应该继续裸跑普通 DPO，而应该加入 **winner anchor + DPOP/Reg-DPO/APO 风格正则化**，防止目标函数只通过 loser 侧退化获得高 `implicit_acc`。第四个实验还额外说明：问题不只是 DiffuEraser 的实现问题，也和 VideoInpainting pair 的局部差异、mask 区域质量、winner/loser 可分性有关。

## 2. Experiment Contents and Per-Experiment Summaries

### 2.1 普通 DiffDPO_loss

**Log:** `/home/hj/log/普通DiffDPO_loss.log`

**实验内容：** 这是 DiffuEraser / Video Inpainting DPO 的普通 baseline。训练目标保留标准 DPO 排序项，winner 和 loser 两侧都参与偏好差值计算；当前分析窗口使用 `global-step<=10000`，诊断来自 `DPO Diagnostics @ Step ...`，并且有 validation PSNR/SSIM。

**关键结果：**

- `dpo_loss` 从 `{diffdpo_delta['l_dpo_first']:.6f}` 降到 `{diffdpo_delta['l_dpo_last']:.6f}`，DPO 排序目标很快被优化到接近 0。
- `implicit_acc` 从 `{diffdpo_delta['implicit_acc_first']:.6f}` 升到 `{diffdpo_delta['implicit_acc_last']:.6f}`，最后 20% 均值为 `{diffdpo_last['implicit_acc']:.6f}`。
- `win_gap` 从 `{diffdpo_delta['win_gap_first']:.6f}` 增大到 `{diffdpo_delta['win_gap_last']:.6f}`；`lose_gap` 从 `{diffdpo_delta['lose_gap_first']:.6f}` 增大到 `{diffdpo_delta['lose_gap_last']:.6f}`。
- 最后 20% 的 `lose_gap - win_gap={diffdpo_last['lose_minus_win_gap']:.6f}`，`loser_dominant_ratio={diffdpo_last['loser_dominant_ratio']:.6f}`。
- PSNR/SSIM 从 `{diffdpo_delta['psnr_first']:.4f} / {diffdpo_delta['ssim_first']:.4f}` 下降到 `{diffdpo_delta['psnr_last']:.4f} / {diffdpo_delta['ssim_last']:.4f}`。

**实验总结：** 这个实验证明裸 DPO 可以非常快地满足相对偏好排序，但满足方式是危险的：policy 并没有稳定改善 winner，反而同时拉高 winner/loser 的误差，并且更强地拉高 loser 误差。`implicit_acc=1` 和 `dpo_loss≈0` 在这里不是质量提升信号，而是 loser shortcut 的表征；PSNR/SSIM 的下降说明这个 shortcut 已经转化为真实 inpainting 质量退化。

### 2.2 删除 lose_gap 的 DiffDPO ablation

**Log:** `/home/hj/log/把lose_gap删除的loss.log`

**实验内容：** 这是 DiffDPO 的 ablation，训练 loss 中删除 loser gap 相关贡献，只保留 winner 侧约束；但日志仍然继续计算 `ml/mrefl/lose_gap` 作为 monitor-only 指标，用来观察 policy 对 loser 分支的副作用。分析窗口同样使用 `global-step<=10000`，并有 validation PSNR/SSIM。

**关键结果：**

- `dpo_loss` 基本保持在 `{no_lose_delta['l_dpo_first']:.6f} -> {no_lose_delta['l_dpo_last']:.6f}`，没有像普通 DiffDPO 那样塌到 0。
- 最后 20% 的 `win_gap={no_lose_last['win_gap']:.6f}`，仍然小于 0，说明 winner 侧没有被推坏。
- 最后 20% 的 `lose_gap={no_lose_last['lose_gap']:.6f}`，`lose_gap - win_gap={no_lose_last['lose_minus_win_gap']:.6f}`，远小于普通 DiffDPO 的 loser-dominant 幅度。
- PSNR/SSIM 从 `{no_lose_delta['psnr_first']:.4f} / {no_lose_delta['ssim_first']:.4f}` 到 `{no_lose_delta['psnr_last']:.4f} / {no_lose_delta['ssim_last']:.4f}`，基本稳定。

**实验总结：** 删除 loser gap 后，winner reconstruction 本身可以稳定工作，质量指标也没有崩。这说明问题不是数据完全不可学，也不是 winner 分支天然坏掉；问题主要来自普通 DPO 允许模型通过扩大 loser 误差来赢。这个实验给后续正则化方向提供了最重要的对照：应该保留偏好学习，但必须给 winner 侧加 anchor，并限制 loser-dominant shortcut。

### 2.3 开源 VideoDPO 原始训练日志

**Log:** `/home/hj/log/VideoDPO的训练.log`

**实验内容：** 这是开源 VideoDPO 代码的训练日志。日志每个 optimizer `global_step` 输出 `[dpo_diag]`，本报告直接使用 `global_step` 作为横坐标，范围为 `{int(video_delta['first_step'])}-{int(video_delta['last_step'])}`。该日志没有 Video Inpainting 的 PSNR/SSIM validation，因此只能比较 DPO 中间指标。

**关键结果：**

- `dpo_loss` 从 `{video_delta['l_dpo_first']:.6f}` 降到 `{video_delta['l_dpo_last']:.6f}`，但不像普通 DiffDPO 那样长期贴 0。
- `implicit_acc` 从 `{video_delta['implicit_acc_first']:.6f}` 到 `{video_delta['implicit_acc_last']:.6f}`；最后 20% 均值为 `{video_last['implicit_acc']:.6f}`。
- 最后 20% 的 `win_gap={video_last['win_gap']:.6f}`，`lose_gap={video_last['lose_gap']:.6f}`，二者为正但只有 1e-3 量级。
- `loser_dominant_ratio` 从前 20% 的 `0.736759` 上升到最后 20% 的 `{video_last['loser_dominant_ratio']:.6f}`。

**实验总结：** 开源 VideoDPO 的 DPO 训练并不要求每个 winner denoising MSE 都优于 reference；后期 `win_gap` 也可以略大于 0。关键区别是它的 gap 漂移量级很小，而且原始 VideoDPO 关注最终 sampled video 的偏好质量，不是单步 epsilon MSE 本身。因此它不能直接证明我们的 inpainting DPO 是安全的。它更像一个参考：VideoDPO 的相对偏好优化会自然产生小幅正 gap，但普通 DiffDPO 的 gap 扩大到了 1e-1 到 1e0，并且伴随 PSNR/SSIM 崩溃，这已经不是同一类现象。

### 2.4 VideoDPO 使用 VideoInpainting 数据

**Log:** `/home/hj/log/使用VideoInpainting的数据集的VideoDPO的loss.log`

**实验内容：** 这是把开源 VideoDPO 训练代码适配到 Video Inpainting winner/loser pair 后得到的日志。winner 主要来自 GT / clean target，loser 来自 inpainting 模型输出。最新日志解析到 `global_step={int(video_inpaint_delta['last_step'])}`：其中 `[dpo_diag]` 每 300 个 global-step 输出一次完整指标，Lightning 进度条则提供更密的 `dpo_loss / implicit_acc / win_gap / lose_gap`。PSNR/SSIM validation 曾尝试运行，但旧 H20 环境缺少 `skimage`，因此没有成功写出质量指标。

**关键结果：**

- `dpo_loss` 从起点 `{video_inpaint_delta['l_dpo_first']:.6f}` 到最新单点 `{video_inpaint_delta['l_dpo_last']:.6f}`；最后 20% 均值为 `{video_inpaint_last['l_dpo']:.6f}`，说明偏好目标整体被压低，但尾部仍存在 batch-level spike。
- `implicit_acc` 从 `{video_inpaint_delta['implicit_acc_first']:.6f}` 到 `{video_inpaint_delta['implicit_acc_last']:.6f}`；最后 20% 均值为 `{video_inpaint_last['implicit_acc']:.6f}`。
- `win_gap` 从 `{video_inpaint_delta['win_gap_first']:.6f}` 增大到 `{video_inpaint_delta['win_gap_last']:.6f}`；`lose_gap` 从 `{video_inpaint_delta['lose_gap_first']:.6f}` 增大到 `{video_inpaint_delta['lose_gap_last']:.6f}`。
- 最后 20% 的 `lose_gap - win_gap={video_inpaint_last['lose_minus_win_gap']:.6f}`，`loser_dominant_ratio={video_inpaint_last['loser_dominant_ratio']:.6f}`。
- validation failure 明确记录为 `ModuleNotFoundError: No module named 'skimage'`，所以这个实验目前仍没有可用 PSNR/SSIM。

**实验总结：** 统一到 Video Inpainting 数据后，开源 VideoDPO 也出现了明显的 loser-dominant 倾向：`lose_gap` 大多数时间高于 `win_gap`，并且 `[dpo_diag]` 给出的 `loser_dominant_ratio` 后期接近 1。这个结果说明风险不只是 DiffuEraser 某段实现的偶然问题，而是 Video Inpainting pair 的局部差异和普通 DPO 相对排序目标之间存在结构性冲突。由于最新日志仍没有 PSNR/SSIM，这个实验不能单独判断最终视觉质量，但它已经足够说明：只靠裸 DPO 排序，不足以保证 inpainting 质量会变好。

## 3. Experiment Scope and Step Setting

主对比使用如下数据窗口：

{fmt_df(scope_df)}

关键点：

- Lightning 进度条里的 `Epoch 0: 0/1785` 只是 dataloader 的 batch 进度，不作为本报告横坐标。
- 本报告横坐标统一叫 `global-step`，含义是训练框架里的 optimizer `global_step`。
- 对 DiffDPO 两个日志，原始 `DPO Diagnostics @ Step` 按 global-step 使用；对 VideoDPO 日志，优先读取 `[dpo_diag] global_step=...`，并把 Lightning 进度条还原为 `epoch * steps_per_epoch + batch_idx` 后合并到同一 global-step 轴。
- 完整解析结果保存在 `{table_files['all_diagnostics_full']}`；主图和主表使用 `{table_files['all_diagnostics']}`。

## 4. Metric Definitions

当前诊断口径：

- `mw` / `mse_w`: policy 在 winner/GT 样本上的 epsilon MSE。
- `ml` / `mse_l`: policy 在 loser/model-output 样本上的 epsilon MSE。
- `mrefw` / `ref_mse_w`: reference 在 winner 上的 epsilon MSE。
- `mrefl` / `ref_mse_l`: reference 在 loser 上的 epsilon MSE。
- `win_gap = mw - mrefw`。小于 0 表示 policy 在 winner denoising MSE 上优于 ref。
- `lose_gap = ml - mrefl`。大于 0 表示 policy 在 loser denoising MSE 上差于 ref。
- 普通 DPO 的核心判断近似是 `win_gap < lose_gap`，而不是要求 `win_gap < 0`。
- `implicit_acc` 是当前 batch / 当前 timestep / 当前 noise 上 `inside_term > 0` 的比例，分布式时会 gather 所有 rank；它不是累计平均。

重要解释：`implicit_acc` 高只说明相对排序被满足，不说明 winner 质量变好。普通 DiffDPO 的日志已经证明了这一点：`implicit_acc` 可以接近 1，同时 PSNR/SSIM 大幅下降。

## 5. Phase Summary

每个实验取诊断记录的前 20% 和后 20% 做均值：

{fmt_df(phase_df)}

## 6. Start-to-End Delta

{fmt_df(delta_df)}

## 7. Validation-Aligned Table

DiffuEraser 两个实验有 PSNR/SSIM。开源 VideoDPO 日志没有 PSNR/SSIM。第四个 VideoDPO+Inpainting 实验尝试了 `video_inpaint_val`，但 H20 的 VideoDPO 环境缺少 `skimage`，所以没有成功写出 PSNR/SSIM；这意味着目前不能把第四个实验和前两个实验做质量指标的公平数值对比，只能比较中间 DPO diagnostic。

{fmt_df(validation_df)}

Validation failure 记录：

{failure_text}

## 8. VideoDPO 300-Step Loss and Metric Changes

开源 VideoDPO 每一步都有 `[dpo_diag]`，主图和这里的表格都按 300 step 聚合，避免逐 step outlier 把 y 轴拉爆。第四个实验现在同时包含每 300 step 的 `[dpo_diag]` 和更密的 Lightning 进度条指标；这里也按同一粒度聚合展示：

{fmt_df(chunks_df)}

## 9. Experiment Interpretation

### 9.1 普通 DiffDPO_loss

最后 20%：

- `implicit_acc={diffdpo_last['implicit_acc']:.6f}`
- `win_gap={diffdpo_last['win_gap']:.6f}`
- `lose_gap={diffdpo_last['lose_gap']:.6f}`
- `lose_gap - win_gap={diffdpo_last['lose_minus_win_gap']:.6f}`
- `loser_dominant_ratio={diffdpo_last['loser_dominant_ratio']:.6f}`

这个实验的问题不是指标错了，而是裸 DPO 目标可以被“把 loser 变差更多”轻易满足。PSNR/SSIM 同步下降说明 policy 的真实 inpainting 质量也在退化。

### 9.2 删除 lose_gap 的 ablation

最后 20%：

- `implicit_acc={no_lose_last['implicit_acc']:.6f}`
- `win_gap={no_lose_last['win_gap']:.6f}`
- `lose_gap={no_lose_last['lose_gap']:.6f}`
- `PSNR/SSIM` 基本稳定在 24.19 / 0.866 附近

虽然 loss 中删除了 loser 项，`ml/mrefl/lose_gap` 仍然可以作为 **monitor-only metrics** 计算。它们的意义是观察 policy 对 loser 的副作用，而不是训练信号。这个 ablation 的目的就是确认 winner branch 是否能单独 work；结果是可以。

### 9.3 开源 VideoDPO

VideoDPO 的 `win_gap` 后期略大于 0，但量级很小，且它的优化目标本来就是最终视频偏好，不是最小化每个 winner 的 epsilon MSE。这个现象说明：

- `win_gap > 0` 不能直接推出最终视频一定不如 ref。
- 但在我们的 DiffuEraser 日志里，`win_gap/lose_gap` 的正向漂移是 1e-1 到 1e0 量级，并且 PSNR/SSIM 真实下降，所以是实质性退化。

### 9.4 VideoDPO 使用 VideoInpainting 数据

主窗口最后 20%：

- `implicit_acc={video_inpaint_last['implicit_acc']:.6f}`
- `win_gap={video_inpaint_last['win_gap']:.6f}`
- `lose_gap={video_inpaint_last['lose_gap']:.6f}`
- `lose_gap - win_gap={video_inpaint_last['lose_minus_win_gap']:.6f}`
- `loser_dominant_ratio={video_inpaint_last['loser_dominant_ratio']:.6f}`

这个实验的重点不是“VideoDPO 代码一定坏了”，而是统一成 VideoInpainting 数据后，VideoDPO 也会很快把偏好目标做成 `lose_gap > win_gap`。而且 `loser_dominant_ratio` 在主窗口内几乎一直为 1，说明排序胜利高度依赖 loser 侧变差。它和普通 DiffDPO 的失败方向是一致的，只是数值尺度和 log 频率不同。

第四个实验的 PSNR/SSIM 没有成功记录，原因不是逻辑上没有验证，而是日志里明确报了：

```text
[video_inpaint_val] step=2000 failed: ModuleNotFoundError: No module named 'skimage'
[video_inpaint_val] step=4000 failed: ModuleNotFoundError: No module named 'skimage'
[video_inpaint_val] step=6000 failed: ModuleNotFoundError: No module named 'skimage'
```

因此第四个实验目前只能用于判断 DPO 中间指标趋势；如果要比较最终 inpainting 质量，需要先修 H20 环境的 `skimage` 或把 metric import 改成项目里已有且不依赖 `skimage` 的实现。

## 10. Why Inpainting Preference Pairs Are Harder

你这个判断是对的：Video Inpainting 的偏好对和主流 T2V VideoDPO 不完全一样。

主流 VideoDPO 常见数据：同一个 prompt 通过 T2V 生成多个候选，winner/loser 可能在外观、构图、运动、对象状态上差异很大。相对排序信号更粗，但更容易被 DPO 区分。

我们的 Video Inpainting 数据：

- winner 多来自 GT。
- loser 来自某个 inpainting 模型输出。
- 背景和非 mask 区域大量相同。
- 真正差异主要在 mask 区域，而且常见问题是 blur、artifact、flicker、temporal inconsistency。

这会带来两个问题：

1. **epsilon MSE 可能被大面积相同区域稀释。** 如果 loss/diagnostic 没有足够 mask-aware 或 temporal-aware，模型会看到一个很弱、很局部的偏好信号。
2. **loser 更容易成为捷径。** 因为 loser 的缺陷可能只在 mask 区域，而全局 DPO 只要求相对差距，模型可以通过把 loser 区域进一步弄差来获得高 `implicit_acc`，而不是真正改善 winner/GT reconstruction。

所以 `implicit_acc` 的难易程度确实和数据集 pair separability 有关，但不只由难易程度决定；还受 mask 覆盖、噪声 timestep、beta、loss 权重、ref/policy 初始距离、MSE 是否能捕捉 flicker 等因素影响。对当前任务，`implicit_acc` 必须和 `win_gap`、`lose_gap`、`loser_dominant_ratio`、PSNR/SSIM 一起看。

## 11. Proposed Regularized Objective

建议下一轮采用你图里的方向：

```text
L_total =
  L_DPO_norm
  + lambda_a * m_w
  + lambda_w * ReLU(m_w - m_w_ref)
  + lambda_g * ReLU(tilde_lose_gap - tilde_win_gap - tau_g)
```

含义：

- 当前四个实验的主表和图只展示 `DPO_loss`，因为现有配置里正则权重为 0，`Total_loss` 基本等于 `DPO_loss`。
- 只有下一轮真正加入正则项之后，`Total_loss` 才需要单独进入图表。
- `L_DPO_norm`: 保留偏好排序，但建议对 gap 做 normalization，避免不同 timestep / mask 面积造成尺度漂移。
- `lambda_a * m_w`: Reg-DPO / SFT-style positive anchor，直接约束 winner/GT 分支不能飘。
- `lambda_w * ReLU(m_w - m_w_ref)`: DPOP/Smaug-style positive protection。只有当 policy 在 winner 上比 ref 更差时才惩罚。
- `lambda_g * ReLU(tilde_lose_gap - tilde_win_gap - tau_g)`: anti-loser-dominance。允许 loser 有一定 margin，但当模型主要靠扩大 loser gap 获胜时惩罚。

### Recommended First Sweep

建议先做保守 sweep：

| run | lambda_a | lambda_w | lambda_g | tau_g | note |
|---|---:|---:|---:|---:|---|
| R1 | 0.01 | 1.0 | 0.0 | - | 只验证 winner anchor 是否稳住 PSNR |
| R2 | 0.01 | 1.0 | 0.1 | 0.05 | 加轻量 anti-loser-dominance |
| R3 | 0.05 | 1.0 | 0.1 | 0.05 | 更强 winner reconstruction |
| R4 | 0.01 | 5.0 | 0.1 | 0.05 | 更强 DPOP-style ref anchor |

### Stop / Save Criteria

不要再用 `implicit_acc` 单独判断成功。建议采用：

- 必须：`PSNR/SSIM` 不低于 no-lose-gap ablation 的稳定水平太多。
- 必须：`win_gap` 不能长期大幅为正。
- 必须：`lose_gap - win_gap` 不能持续放大到普通 DiffDPO 的量级。
- 参考：`implicit_acc` 处在 0.55-0.9 可接受；如果过快到 1.0，要检查是否又出现 loser shortcut。

## 12. Paper Connection

- `Smaug.pdf / DPOP`: 指出普通 DPO 可以在 preferred likelihood 下降的情况下仍然提升相对偏好；对应我们这里的 `win_gap > 0` 风险。解决思路是给 winner/preferred 加 positive protection。
- `Reg-DPO_compressed.pdf`: 指出视频 DPO 容易通过拉大正负样本误差差距获得低 loss，并加入 SFT regularization 稳住正样本；这和普通 DiffDPO 的 PSNR/SSIM collapse 高度一致。
- `Anchored Preference Optimization and Contrastive Revisions.pdf`: 指出 preference objective 缺少绝对锚点，数据 pair 如果不够 contrastive 会导致信用分配混乱；对 inpainting 的局部 mask 差异尤其相关。

## 13. Files

- Phase summary: `{table_files['phase_summary']}`
- Delta summary: `{table_files['delta_summary']}`
- Validation-aligned metrics: `{table_files['validation_aligned']}`
- VideoDPO 300-step chunk metrics: `{table_files['videodpo_300step_chunks']}`
- All parsed diagnostics: `{table_files['all_diagnostics']}`
- Full raw diagnostics before analysis cutoff: `{table_files['all_diagnostics_full']}`
- Experiment scope: `{table_files['experiment_scope']}`
- Validation failures: `{table_files['validation_failures']}`
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    diags: dict[str, pd.DataFrame] = {}
    vals: dict[str, pd.DataFrame] = {}
    all_diag_frames = []
    all_diag_full_frames = []
    phase_frames = []
    delta_rows = []
    validation_frames = []
    validation_failure_frames = []
    chunk_frames = []
    scope_rows = []

    for key, cfg in LOGS.items():
        label = cfg["label"]
        if cfg["kind"] == "diffueraser":
            raw_diag, val = parse_diffueraser(cfg["path"])
            val_fail = pd.DataFrame()
        else:
            raw_diag, val, val_fail = parse_videodpo(cfg["path"])

        raw_diag_with_label = raw_diag.copy()
        raw_diag_with_label.insert(0, "experiment", label)
        all_diag_full_frames.append(raw_diag_with_label)
        raw_diag_with_label.to_csv(OUT_DIR / f"{key}_diagnostics_full.csv", index=False)

        diag = raw_diag.copy()
        cutoff = cfg.get("analysis_cutoff_step")
        if cutoff is not None:
            diag = diag[diag["step"] <= cutoff].copy()
            val = val[val["step"] <= cutoff].copy() if not val.empty else val

        scope_rows.append(
            {
                "experiment": label,
                "raw_rows": len(raw_diag),
                "used_rows": len(diag),
                "raw_step_range": (
                    f"{int(raw_diag.step.min())}-{int(raw_diag.step.max())}" if not raw_diag.empty else ""
                ),
                "used_step_range": f"{int(diag.step.min())}-{int(diag.step.max())}" if not diag.empty else "",
                "analysis_cutoff_step": cutoff if cutoff is not None else "",
                "analysis_note": cfg.get("analysis_note", "full log is used"),
            }
        )

        diag.insert(0, "experiment", label)
        val.insert(0, "experiment", label) if not val.empty else None
        if not val_fail.empty:
            val_fail.insert(0, "experiment", label)
            validation_failure_frames.append(val_fail)
        diags[key] = diag.drop(columns=["experiment"], errors="ignore")
        vals[key] = val.drop(columns=["experiment"], errors="ignore") if not val.empty else val

        all_diag_frames.append(diag)
        phase_frames.append(phase_summary(diag.drop(columns=["experiment"], errors="ignore"), label))
        delta_rows.append(delta_summary(diag.drop(columns=["experiment"], errors="ignore"), val.drop(columns=["experiment"], errors="ignore") if not val.empty else val, label))
        if not val.empty:
            validation_frames.append(nearest_diag_for_vals(diag.drop(columns=["experiment"], errors="ignore"), val.drop(columns=["experiment"], errors="ignore"), label))
        if key.startswith("videodpo"):
            chunk_frames.append(chunk_summary(diag.drop(columns=["experiment"], errors="ignore"), label, chunk=300))

        diag.to_csv(OUT_DIR / f"{key}_diagnostics.csv", index=False)
        if not val.empty:
            val.to_csv(OUT_DIR / f"{key}_validation.csv", index=False)

    phase_df = pd.concat(phase_frames, ignore_index=True)
    delta_df = pd.DataFrame(delta_rows)
    validation_df = pd.concat(validation_frames, ignore_index=True) if validation_frames else pd.DataFrame()
    validation_failures_df = (
        pd.concat(validation_failure_frames, ignore_index=True) if validation_failure_frames else pd.DataFrame()
    )
    chunks_df = pd.concat(chunk_frames, ignore_index=True) if chunk_frames else pd.DataFrame()
    all_diag_df = pd.concat(all_diag_frames, ignore_index=True)
    all_diag_full_df = pd.concat(all_diag_full_frames, ignore_index=True)
    scope_df = pd.DataFrame(scope_rows)

    phase_df.to_csv(OUT_DIR / "phase_summary.csv", index=False)
    delta_df.to_csv(OUT_DIR / "delta_summary.csv", index=False)
    validation_df.to_csv(OUT_DIR / "validation_aligned.csv", index=False)
    validation_failures_df.to_csv(OUT_DIR / "validation_failures.csv", index=False)
    chunks_df.to_csv(OUT_DIR / "videodpo_300step_chunks.csv", index=False)
    all_diag_df.to_csv(OUT_DIR / "all_diagnostics.csv", index=False)
    all_diag_full_df.to_csv(OUT_DIR / "all_diagnostics_full.csv", index=False)
    scope_df.to_csv(OUT_DIR / "experiment_scope.csv", index=False)

    plot_all(diags, vals)
    DOC_PATH.write_text(
        build_doc(diags, vals, phase_df, delta_df, validation_df, chunks_df, scope_df, validation_failures_df)
    )

    print(f"Wrote {DOC_PATH}")
    print(f"Wrote artifacts under {OUT_DIR}")
    print("Parsed rows:")
    for key, df in diags.items():
        raw_len = len(all_diag_full_df[all_diag_full_df["experiment"] == LOGS[key]["label"]])
        print(
            f"  {LOGS[key]['label']}: diagnostics={len(df)} used/{raw_len} raw, "
            f"validation={len(vals[key]) if vals[key] is not None else 0}"
        )


if __name__ == "__main__":
    main()
