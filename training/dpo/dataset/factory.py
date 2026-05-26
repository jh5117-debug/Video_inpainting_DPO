"""Dataset factory for DPO experiments."""

from __future__ import annotations

from training.dpo.dataset.dpo_dataset import DPODataset
from training.dpo.dataset.generated_loser_manifest_dataset import GeneratedLoserManifestDataset
from training.dpo.dataset.videodpo_fullmask_dataset import VideoDPOFullMaskDiffuEraserDataset


def build_dpo_dataset(args, tokenizer):
    dataset_type = getattr(args, "dpo_dataset_type", "diffueraser_inpainting")
    if dataset_type == "diffueraser_inpainting":
        return DPODataset(args, tokenizer, dpo_data_root=args.dpo_data_root)
    if dataset_type == "videodpo_fullmask":
        return VideoDPOFullMaskDiffuEraserDataset(args, tokenizer, dpo_data_root=args.dpo_data_root)
    if dataset_type == "generated_loser_manifest":
        return GeneratedLoserManifestDataset(args, tokenizer)
    raise ValueError(f"Unknown --dpo_dataset_type: {dataset_type}")
