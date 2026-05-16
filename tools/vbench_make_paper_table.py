#!/usr/bin/env python
"""Create a VideoDPO-paper-style table from VBench summary files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_LABEL_MAP = {
    "vc2_base": "Baseline[7]",
    "vc2_dpo": "VideoDPO",
}


def parse_label_map(value: str) -> dict[str, str]:
    mapping = dict(DEFAULT_LABEL_MAP)
    if not value:
        return mapping
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Bad label map item: {item!r}")
        key, name = item.split("=", 1)
        mapping[key.strip()] = name.strip()
    return mapping


def parse_labels(ckpt_specs: str | None, out_root: Path) -> list[str]:
    if ckpt_specs:
        labels = []
        for item in ckpt_specs.split(","):
            item = item.strip()
            if not item:
                continue
            labels.append(item.split(":", 1)[0])
        if labels:
            return labels
    return sorted(
        p.name for p in out_root.iterdir() if (p / "vbench_eval" / "summary.json").is_file()
    )


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def load_row(out_root: Path, label: str, model_group: str, label_map: dict[str, str]) -> dict[str, str]:
    summary_path = out_root / label / "vbench_eval" / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing summary for {label}: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    return {
        "Backbone": model_group,
        "Model": label_map.get(label, label),
        "VBench Total (%)": fmt(summary.get("total_score_percent")),
        "VBench Quality (%)": fmt(summary.get("quality_score_percent")),
        "VBench Semantics (%)": fmt(summary.get("semantic_score_percent")),
        "HPS (V)": "-",
        "PickScore": "-",
        "label": label,
        "summary_json": str(summary_path),
    }


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    columns = [
        "Backbone",
        "Model",
        "VBench Total (%)",
        "VBench Quality (%)",
        "VBench Semantics (%)",
        "HPS (V)",
        "PickScore",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[col] for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Backbone & Model & Total & Quality & Semantics & HPS(V) & PickScore \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['Backbone']} & {row['Model']} & {row['VBench Total (%)']} & "
            f"{row['VBench Quality (%)']} & {row['VBench Semantics (%)']} & "
            f"{row['HPS (V)']} & {row['PickScore']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--ckpt_specs", default="")
    parser.add_argument("--model_group", default="VC2")
    parser.add_argument("--label_map", default="")
    parser.add_argument("--output_prefix", type=Path, required=True)
    args = parser.parse_args()

    label_map = parse_label_map(args.label_map)
    labels = parse_labels(args.ckpt_specs, args.out_root)
    rows = [load_row(args.out_root, label, args.model_group, label_map) for label in labels]

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    md_path = args.output_prefix.with_suffix(".md")
    tex_path = args.output_prefix.with_suffix(".tex")

    fieldnames = [
        "Backbone",
        "Model",
        "VBench Total (%)",
        "VBench Quality (%)",
        "VBench Semantics (%)",
        "HPS (V)",
        "PickScore",
        "label",
        "summary_json",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(md_path, rows)
    write_latex(tex_path, rows)

    print(f"[paper-table] wrote {csv_path}")
    print(f"[paper-table] wrote {md_path}")
    print(f"[paper-table] wrote {tex_path}")
    print(md_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
