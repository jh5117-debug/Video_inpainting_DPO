#!/usr/bin/env python3
"""Exp16 implementation preflight.

This preflight blocks early unless a manifest with real ProPainter prior paths
is available. It also runs tensor-level x0/confidence/loss sanity checks so
helpers fail loudly before any long training is launched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from exp16_dataset import audit_manifest_for_prior
from exp16_loss import (
    PriorConfidenceConfig,
    compute_prior_confidence_from_gt_error,
    compute_prior_gated_losses,
    predict_x0_from_model_output,
)


class DummyScheduler:
    def __init__(self) -> None:
        self.alphas_cumprod = torch.linspace(0.999, 0.001, 1000)
        self.config = type("Config", (), {"prediction_type": "epsilon"})()


def tensor_sanity() -> dict[str, float | str]:
    torch.manual_seed(7)
    b, f, c, h, w = 1, 2, 3, 32, 48
    gt = torch.rand(b, f, c, h, w) * 2 - 1
    prior = (gt + 0.05 * torch.randn_like(gt)).clamp(-1, 1)
    brushnet_masks = torch.ones(b, f, 1, h, w)
    brushnet_masks[:, :, :, 8:24, 12:36] = 0.0
    hole = 1.0 - brushnet_masks
    conf, conf_stats = compute_prior_confidence_from_gt_error(prior, gt, hole, alpha=5.0)

    n = b * f
    latent_h, latent_w = 8, 12
    z_gt = torch.randn(n, 4, latent_h, latent_w)
    z_prior = z_gt + 0.03 * torch.randn_like(z_gt)
    noise = torch.randn_like(z_gt)
    scheduler = DummyScheduler()
    timesteps = torch.tensor([10, 10])
    noisy = scheduler.alphas_cumprod[timesteps].reshape(-1, 1, 1, 1).sqrt() * z_gt
    noisy = noisy + (1.0 - scheduler.alphas_cumprod[timesteps]).reshape(-1, 1, 1, 1).sqrt() * noise
    model_output = noise + 0.01 * torch.randn_like(noise)
    z_hat = predict_x0_from_model_output(noisy, model_output, timesteps, scheduler)
    extra, loss_stats = compute_prior_gated_losses(
        z_hat,
        z_prior,
        z_gt,
        brushnet_masks,
        conf,
        PriorConfidenceConfig(),
    )
    if not torch.isfinite(extra):
        raise RuntimeError("Exp16 tensor sanity loss is not finite")
    out: dict[str, float | str] = {"tensor_sanity": "passed", "extra_loss": float(extra.detach())}
    out.update(conf_stats)
    out.update(loss_stats)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report_json", default="exp16_prior_confidence_gated_dpo/reports/preflight_report.json")
    parser.add_argument("--report_md", default="reports/exp16_preflight_report.md")
    parser.add_argument("--limit_audit", type=int, default=200)
    args = parser.parse_args()

    manifest_audit = audit_manifest_for_prior(args.manifest, limit=args.limit_audit)
    tensor = tensor_sanity()
    status = "passed" if manifest_audit.get("with_prior", 0) == manifest_audit.get("total", 0) and manifest_audit.get("total", 0) > 0 else "blocked"
    blocked_reason = ""
    if status != "passed":
        blocked_reason = "manifest lacks real ProPainter prior paths; build exp16 prior cache first"

    report = {
        "status": status,
        "blocked_reason": blocked_reason,
        "manifest_audit": manifest_audit,
        "tensor_sanity": tensor,
        "x0_prior_loss_proxy": False,
    }

    report_json = Path(args.report_json)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_md = Path(args.report_md)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Exp16 Preflight Report",
        "",
        f"status: `{status}`",
        f"blocked_reason: `{blocked_reason}`" if blocked_reason else "blocked_reason: ``",
        "",
        "## Manifest Audit",
        "",
        f"- manifest: `{manifest_audit.get('manifest')}`",
        f"- exists: {manifest_audit.get('exists')}",
        f"- total audited rows: {manifest_audit.get('total')}",
        f"- rows with prior: {manifest_audit.get('with_prior')}",
        f"- missing prior: {manifest_audit.get('missing_prior')}",
        "",
        "## Tensor Sanity",
        "",
        f"- tensor sanity: `{tensor.get('tensor_sanity')}`",
        f"- extra loss: {tensor.get('extra_loss')}",
        f"- prior_conf_mean: {tensor.get('prior_conf_mean')}",
        f"- L_prior: {tensor.get('L_prior')}",
        f"- L_gen: {tensor.get('L_gen')}",
        f"- L_boundary_extra: {tensor.get('L_boundary_extra')}",
        "",
        "This preflight does not launch training. Exp16 training remains blocked",
        "until every row has a real ProPainter prior path and the full trainer",
        "integrates `z_hat_x0`, `z_prior`, and `z_gt` into total loss.",
    ]
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
