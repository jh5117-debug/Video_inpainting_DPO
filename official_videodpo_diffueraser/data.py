from __future__ import annotations

from types import SimpleNamespace

from transformers import AutoTokenizer

from training.dpo.dataset.videodpo_fullmask_dataset import VideoDPOFullMaskDiffuEraserDataset


class OfficialVideoDPOFullMaskDataset(VideoDPOFullMaskDiffuEraserDataset):
    """VideoDPO VC2 pairs emitted in the DiffuEraser full-mask batch contract.

    The class keeps the official VideoDPO ``DataModuleFromConfig`` entrypoint
    usable: official ``scripts/train.py`` instantiates this target directly from
    YAML, while the heavy lifting stays in the project's vetted full-mask
    dataset adapter.
    """

    def __init__(
        self,
        data_root: str,
        base_model_name_or_path: str,
        tokenizer_name: str | None = None,
        resolution: int | list[int] = 512,
        train_height: int | None = 320,
        train_width: int | None = 512,
        video_length: int = 16,
        frame_stride: int = 1,
        clip_length: float = 1.0,
        full_mask_value: float = 0.0,
        proportion_empty_prompts: float = 0.0,
        seed: int = 42,
        max_resample_attempts: int = 64,
        **_,
    ):
        if isinstance(resolution, (list, tuple)):
            train_height = int(resolution[0])
            train_width = int(resolution[1])
            resolution = max(int(train_height), int(train_width))

        tokenizer_source = tokenizer_name or base_model_name_or_path
        tokenizer_kwargs = {"revision": None, "use_fast": False}
        if tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source, subfolder="tokenizer", **tokenizer_kwargs
            )

        args = SimpleNamespace(
            dpo_data_root=data_root,
            nframes=int(video_length),
            resolution=int(resolution),
            train_height=train_height,
            train_width=train_width,
            videodpo_frame_stride=int(frame_stride),
            videodpo_clip_length=float(clip_length),
            videodpo_full_mask_value=float(full_mask_value),
            proportion_empty_prompts=float(proportion_empty_prompts),
            seed=int(seed),
            max_resample_attempts=int(max_resample_attempts),
        )
        super().__init__(args=args, tokenizer=tokenizer, dpo_data_root=data_root)

