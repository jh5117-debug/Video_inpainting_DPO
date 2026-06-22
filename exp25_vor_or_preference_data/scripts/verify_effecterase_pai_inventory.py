#!/usr/bin/env python3
"""Verify PAI final EffectErase core download inventory."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

try:
    from effecterase_common import DEFAULT_PAI_DOWNLOAD_DIR, DEFAULT_PAI_HOST, DEFAULT_PAI_KEY, REPORTS, RUNTIME, continuity_report, read_json, shquote, ssh_cmd
except ModuleNotFoundError:
    from .effecterase_common import DEFAULT_PAI_DOWNLOAD_DIR, DEFAULT_PAI_HOST, DEFAULT_PAI_KEY, REPORTS, RUNTIME, continuity_report, read_json, shquote, ssh_cmd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default="reports/effecterase_remote_inventory.json")
    parser.add_argument("--pai-host", default=DEFAULT_PAI_HOST)
    parser.add_argument("--pai-key", type=Path, default=DEFAULT_PAI_KEY)
    parser.add_argument("--pai-download-dir", type=Path, default=DEFAULT_PAI_DOWNLOAD_DIR)
    args = parser.parse_args()
    inv = read_json(Path(args.inventory))
    if not inv:
        raise SystemExit("missing inventory")
    revision = inv["revision"]
    rows = []
    partials = []
    bads = []
    ok = True
    for item in inv["required_files"]:
        final = args.pai_download_dir / revision / item["filename"]
        cmd = f"if [ -f {shquote(str(final))} ]; then stat -c %s {shquote(str(final))}; sha256sum {shquote(str(final))} | awk '{{print $1}}'; else echo MISSING; fi"
        cp = subprocess.run(ssh_cmd(args.pai_key, args.pai_host, cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out = cp.stdout.strip().splitlines()
        exists = out and out[0] != "MISSING"
        size = int(out[0]) if exists else 0
        sha = out[1] if exists and len(out) > 1 else ""
        status = "OK" if exists and size == int(item["size"]) else "BAD"
        ok = ok and status == "OK"
        rows.append({"filename": item["filename"], "group": item["group"], "expected_size": item["size"], "actual_size": size, "sha256": sha, "status": status})
    cmd = f"find {shquote(str(args.pai_download_dir / revision))} -name '*.partial' -o -name '.bad.*' 2>/dev/null || true"
    cp = subprocess.run(ssh_cmd(args.pai_key, args.pai_host, cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in cp.stdout.splitlines():
        if line.endswith(".partial"):
            partials.append(line)
        elif ".bad." in line:
            bads.append(line)
    with (REPORTS / "effecterase_pai_inventory_verification.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "group", "expected_size", "actual_size", "sha256", "status"])
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "revision": revision,
        "ok": ok and not partials and not bads,
        "file_count": len(rows),
        "partials": partials,
        "bads": bads,
        "continuity": continuity_report(inv["required_files"]),
    }
    (REPORTS / "effecterase_pai_inventory_verification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
