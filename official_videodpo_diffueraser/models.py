from __future__ import annotations

import gc
from pathlib import Path

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from diffusers import AutoencoderKL, DDPMScheduler
from einops import rearrange, repeat
from transformers import AutoTokenizer, PretrainedConfig

from libs.brushnet_CA import BrushNetModel
from libs.unet_2d_condition import UNet2DConditionModel
from libs.unet_motion_model import UNetMotionModel
from training.dpo.train_stage1 import compute_dpo_loss, forward_stage1_pair_member
from training.dpo.train_stage2 import forward_stage2_pair_member


def _import_text_encoder_class(base_model_name_or_path: str, revision: str | None = None):
    text_encoder_config = PretrainedConfig.from_pretrained(
        base_model_name_or_path, subfolder="text_encoder", revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    if model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    raise ValueError(f"Unsupported text encoder class: {model_class}")


def _dtype(name: str | None) -> torch.dtype:
    name = (name or "fp32").lower()
    if name in {"fp32", "float32", "32"}:
        return torch.float32
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "16"}:
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def _extract_2d_from_motion(
    motion_unet: UNetMotionModel,
    base_model_name_or_path: str,
    revision: str | None = None,
    variant: str | None = None,
) -> UNet2DConditionModel:
    unet_2d = UNet2DConditionModel.from_pretrained(
        base_model_name_or_path, subfolder="unet", revision=revision, variant=variant
    )
    _copy_motion_2d_to_unet2d(motion_unet, unet_2d)
    return unet_2d


def _copy_motion_2d_to_unet2d(src: UNetMotionModel, dst: UNet2DConditionModel) -> None:
    dst.conv_in.load_state_dict(src.conv_in.state_dict())
    dst.time_proj.load_state_dict(src.time_proj.state_dict())
    dst.time_embedding.load_state_dict(src.time_embedding.state_dict())
    for i, down_block in enumerate(src.down_blocks):
        dst.down_blocks[i].resnets.load_state_dict(down_block.resnets.state_dict())
        if hasattr(dst.down_blocks[i], "attentions") and hasattr(down_block, "attentions"):
            dst.down_blocks[i].attentions.load_state_dict(down_block.attentions.state_dict())
        if dst.down_blocks[i].downsamplers and down_block.downsamplers:
            dst.down_blocks[i].downsamplers.load_state_dict(down_block.downsamplers.state_dict())
    for i, up_block in enumerate(src.up_blocks):
        dst.up_blocks[i].resnets.load_state_dict(up_block.resnets.state_dict())
        if hasattr(dst.up_blocks[i], "attentions") and hasattr(up_block, "attentions"):
            dst.up_blocks[i].attentions.load_state_dict(up_block.attentions.state_dict())
        if dst.up_blocks[i].upsamplers and up_block.upsamplers:
            dst.up_blocks[i].upsamplers.load_state_dict(up_block.upsamplers.state_dict())
    dst.mid_block.resnets.load_state_dict(src.mid_block.resnets.state_dict())
    dst.mid_block.attentions.load_state_dict(src.mid_block.attentions.state_dict())
    if src.conv_norm_out is not None:
        dst.conv_norm_out.load_state_dict(src.conv_norm_out.state_dict())
    if hasattr(src, "conv_act") and src.conv_act is not None:
        dst.conv_act.load_state_dict(src.conv_act.state_dict())
    dst.conv_out.load_state_dict(src.conv_out.state_dict())


def _copy_unet2d_to_motion(src: UNet2DConditionModel, dst: UNetMotionModel) -> None:
    dst.conv_in.load_state_dict(src.conv_in.state_dict())
    dst.time_proj.load_state_dict(src.time_proj.state_dict())
    dst.time_embedding.load_state_dict(src.time_embedding.state_dict())
    for i, down_block in enumerate(src.down_blocks):
        dst.down_blocks[i].resnets.load_state_dict(down_block.resnets.state_dict())
        if hasattr(dst.down_blocks[i], "attentions") and hasattr(down_block, "attentions"):
            dst.down_blocks[i].attentions.load_state_dict(down_block.attentions.state_dict())
        if dst.down_blocks[i].downsamplers and down_block.downsamplers:
            dst.down_blocks[i].downsamplers.load_state_dict(down_block.downsamplers.state_dict())
    for i, up_block in enumerate(src.up_blocks):
        dst.up_blocks[i].resnets.load_state_dict(up_block.resnets.state_dict())
        if hasattr(dst.up_blocks[i], "attentions") and hasattr(up_block, "attentions"):
            dst.up_blocks[i].attentions.load_state_dict(up_block.attentions.state_dict())
        if dst.up_blocks[i].upsamplers and up_block.upsamplers:
            dst.up_blocks[i].upsamplers.load_state_dict(up_block.upsamplers.state_dict())
    dst.mid_block.resnets.load_state_dict(src.mid_block.resnets.state_dict())
    dst.mid_block.attentions.load_state_dict(src.mid_block.attentions.state_dict())
    if src.conv_norm_out is not None:
        dst.conv_norm_out.load_state_dict(src.conv_norm_out.state_dict())
    if hasattr(src, "conv_act") and src.conv_act is not None:
        dst.conv_act.load_state_dict(src.conv_act.state_dict())
    dst.conv_out.load_state_dict(src.conv_out.state_dict())


class OfficialVideoDPODiffuEraser(pl.LightningModule):
    """DiffuEraser model adapter hosted by official VideoDPO's train.py.

    ``stage1`` trains UNet2D + BrushNet. ``stage2`` loads Stage1's 2D/BrushNet
    result and trains only the MotionModule, mirroring the vetted DiffuEraser
    DPO implementation while keeping official VideoDPO's Lightning skeleton.
    """

    lora_args: list = []
    empty_paras = None

    def __init__(
        self,
        stage: str,
        base_model_name_or_path: str,
        vae_path: str,
        ref_model_path: str,
        baseline_unet_path: str | None = None,
        pretrained_dpo_stage1: str | None = None,
        revision: str | None = None,
        variant: str | None = None,
        tokenizer_name: str | None = None,
        nframes: int = 16,
        learning_rate: float = 6e-6,
        beta_dpo: float = 5000.0,
        sft_reg_weight: float = 0.0,
        lose_gap_weight: float = 1.0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-8,
        max_grad_norm: float = 1.0,
        forward_dtype: str = "fp32",
        vae_dtype: str = "fp32",
        ref_dtype: str = "fp32",
        text_dtype: str = "fp32",
        gradient_checkpointing: bool = True,
        split_pos_neg_forward: bool = True,
        use_8bit_adam: bool = False,
        logdir: str | None = None,
        **_,
    ):
        super().__init__()
        self.stage = stage.lower()
        if self.stage not in {"stage1", "stage2"}:
            raise ValueError("stage must be 'stage1' or 'stage2'")
        self.base_model_name_or_path = base_model_name_or_path
        self.vae_path = vae_path
        self.ref_model_path = ref_model_path
        self.baseline_unet_path = baseline_unet_path
        self.pretrained_dpo_stage1 = pretrained_dpo_stage1
        self.revision = revision
        self.variant = variant
        self.nframes = int(nframes)
        self.base_learning_rate = float(learning_rate)
        self.learning_rate = float(learning_rate)
        self.beta_dpo = float(beta_dpo)
        self.sft_reg_weight = float(sft_reg_weight)
        self.lose_gap_weight = float(lose_gap_weight)
        self.adam_beta1 = float(adam_beta1)
        self.adam_beta2 = float(adam_beta2)
        self.adam_weight_decay = float(adam_weight_decay)
        self.adam_epsilon = float(adam_epsilon)
        self.max_grad_norm = float(max_grad_norm)
        self.forward_dtype = _dtype(forward_dtype)
        self.vae_forward_dtype = _dtype(vae_dtype)
        self.ref_forward_dtype = _dtype(ref_dtype)
        self.text_forward_dtype = _dtype(text_dtype)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.split_pos_neg_forward = bool(split_pos_neg_forward)
        self.use_8bit_adam = bool(use_8bit_adam)
        self.logdir = logdir
        self.monitor = None

        tokenizer_source = tokenizer_name or base_model_name_or_path
        if tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, revision=revision, use_fast=False)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source, subfolder="tokenizer", revision=revision, use_fast=False
            )
        text_encoder_cls = _import_text_encoder_class(base_model_name_or_path, revision)
        self.noise_scheduler = DDPMScheduler.from_pretrained(base_model_name_or_path, subfolder="scheduler")
        self.text_encoder = text_encoder_cls.from_pretrained(
            base_model_name_or_path, subfolder="text_encoder", revision=revision, variant=variant
        )
        self.vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae", revision=revision, variant=variant)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        if self.stage == "stage1":
            self._init_stage1()
        else:
            self._init_stage2()

        self.vae.to(dtype=self.vae_forward_dtype)
        self.text_encoder.to(dtype=self.text_forward_dtype)
        self.unet_ref.to(dtype=self.ref_forward_dtype)
        self.brushnet_ref.to(dtype=self.ref_forward_dtype)
        self.unet_main.to(dtype=self.forward_dtype)
        self.brushnet.to(dtype=self.forward_dtype)

    def _init_stage1(self) -> None:
        config_path = Path(self.ref_model_path) / "unet_main" / "config.json"
        is_motion = False
        if config_path.exists():
            import json

            is_motion = json.loads(config_path.read_text()).get("_class_name") == "UNetMotionModel"
        if is_motion:
            motion = UNetMotionModel.from_pretrained(self.ref_model_path, subfolder="unet_main")
            self.unet_main = _extract_2d_from_motion(
                motion, self.base_model_name_or_path, self.revision, self.variant
            )
            del motion
            motion_ref = UNetMotionModel.from_pretrained(self.ref_model_path, subfolder="unet_main")
            self.unet_ref = _extract_2d_from_motion(
                motion_ref, self.base_model_name_or_path, self.revision, self.variant
            )
            del motion_ref
        else:
            self.unet_main = UNet2DConditionModel.from_pretrained(self.ref_model_path, subfolder="unet_main")
            self.unet_ref = UNet2DConditionModel.from_pretrained(self.ref_model_path, subfolder="unet_main")

        self.brushnet = BrushNetModel.from_pretrained(self.ref_model_path, subfolder="brushnet")
        self.brushnet_ref = BrushNetModel.from_pretrained(self.ref_model_path, subfolder="brushnet")
        self.unet_ref.requires_grad_(False).eval()
        self.brushnet_ref.requires_grad_(False).eval()
        self.unet_main.train()
        self.brushnet.train()
        if self.gradient_checkpointing:
            self.unet_main.enable_gradient_checkpointing()
            self.brushnet.enable_gradient_checkpointing()

    def _init_stage2(self) -> None:
        if not self.baseline_unet_path:
            raise ValueError("stage2 requires baseline_unet_path")
        if not self.pretrained_dpo_stage1:
            raise ValueError("stage2 requires pretrained_dpo_stage1")

        self.unet_main = UNetMotionModel.from_pretrained(self.baseline_unet_path, subfolder="unet_main")
        stage1_unet = UNet2DConditionModel.from_pretrained(
            self.pretrained_dpo_stage1, subfolder="unet_main", variant=self.variant
        )
        _copy_unet2d_to_motion(stage1_unet, self.unet_main)
        del stage1_unet
        self.brushnet = BrushNetModel.from_pretrained(self.pretrained_dpo_stage1, subfolder="brushnet")
        self.brushnet.requires_grad_(False).eval()
        self.unet_main.freeze_unet2d_params()
        self.unet_main.train()

        self.unet_ref = UNetMotionModel.from_pretrained(self.ref_model_path, subfolder="unet_main")
        self.brushnet_ref = BrushNetModel.from_pretrained(self.ref_model_path, subfolder="brushnet")
        self.unet_ref.requires_grad_(False).eval()
        self.brushnet_ref.requires_grad_(False).eval()
        if self.gradient_checkpointing:
            self.unet_main.enable_gradient_checkpointing()

    def inject_lora(self):
        return None

    def get_input(self, batch, k=None):
        return batch

    def _encode_latents(self, video: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        with torch.no_grad():
            latents = self.vae.encode(
                rearrange(video, "b f c h w -> (b f) c h w").to(dtype=dtype)
            ).latent_dist.sample()
        return latents * self.vae.config.scaling_factor

    def _common_batch_tensors(self, batch):
        pos_latents = self._encode_latents(batch["pixel_values_pos"], self.vae_forward_dtype).to(self.forward_dtype)
        neg_latents = self._encode_latents(batch["pixel_values_neg"], self.vae_forward_dtype).to(self.forward_dtype)

        n_batch = batch["conditioning_pixel_values"].shape[0]
        cond_latents = self._encode_latents(
            batch["conditioning_pixel_values"], self.vae_forward_dtype
        ).to(self.forward_dtype)
        cond_latents = rearrange(cond_latents, "(b f) c h w -> b f c h w", b=n_batch)
        masks = F.interpolate(
            batch["masks"].to(dtype=self.forward_dtype),
            size=(1, pos_latents.shape[-2], pos_latents.shape[-1]),
        )
        brushnet_cond = rearrange(torch.cat([cond_latents, masks], 2), "b f c h w -> (b f) c h w")

        noise = torch.randn_like(pos_latents)
        bsz = pos_latents.shape[0] // self.nframes
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=pos_latents.device
        ).long()
        timesteps_expanded = timesteps.repeat_interleave(self.nframes, dim=0)
        noisy_pos = self.noise_scheduler.add_noise(pos_latents, noise, timesteps_expanded)
        noisy_neg = self.noise_scheduler.add_noise(neg_latents, noise, timesteps_expanded)

        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(batch["input_ids"], return_dict=False)[0]
        encoder_hidden_states = encoder_hidden_states.to(dtype=self.forward_dtype)
        encoder_hidden_states_expanded = rearrange(
            repeat(encoder_hidden_states, "b c d -> b t c d", t=self.nframes),
            "b t c d -> (b t) c d",
        )
        return (
            noisy_pos,
            noisy_neg,
            noise,
            timesteps,
            timesteps_expanded,
            encoder_hidden_states,
            encoder_hidden_states_expanded,
            brushnet_cond,
        )

    def training_step(self, batch, batch_idx):
        (
            noisy_pos,
            noisy_neg,
            noise,
            timesteps,
            timesteps_expanded,
            encoder_hidden_states,
            encoder_hidden_states_expanded,
            brushnet_cond,
        ) = self._common_batch_tensors(batch)

        if self.stage == "stage1":
            model_pred, ref_pred = self._stage1_forward_pair(
                noisy_pos, noisy_neg, timesteps_expanded, encoder_hidden_states_expanded, brushnet_cond
            )
        else:
            model_pred, ref_pred = self._stage2_forward_pair(
                noisy_pos,
                noisy_neg,
                timesteps,
                timesteps_expanded,
                encoder_hidden_states,
                encoder_hidden_states_expanded,
                brushnet_cond,
            )

        loss, diagnostics = compute_dpo_loss(
            model_pred,
            ref_pred,
            noise,
            beta_dpo=self.beta_dpo,
            sft_reg_weight=self.sft_reg_weight if self.stage == "stage1" else 0.0,
            lose_gap_weight=self.lose_gap_weight,
            nframes=self.nframes,
        )
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=1)
        self.log(
            "train/loss_simple", loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True, batch_size=1
        )
        for key in ("implicit_acc", "dpo_loss", "mse_w", "mse_l", "win_gap", "lose_gap", "reward_margin"):
            value = diagnostics.get(key)
            if value is not None:
                self.log(f"train/{key}", float(value), prog_bar=(key in {"implicit_acc", "dpo_loss"}), on_step=True, sync_dist=True, batch_size=1)

        del model_pred, ref_pred
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return loss

    def _stage1_forward_pair(self, noisy_pos, noisy_neg, timesteps_expanded, encoder_hidden_states_expanded, brushnet_cond):
        if self.split_pos_neg_forward:
            with torch.no_grad():
                ref_pos = forward_stage1_pair_member(
                    self.brushnet_ref,
                    self.unet_ref,
                    noisy_pos.to(dtype=self.ref_forward_dtype),
                    timesteps_expanded,
                    encoder_hidden_states_expanded.to(dtype=self.ref_forward_dtype),
                    brushnet_cond.to(dtype=self.ref_forward_dtype),
                    self.ref_forward_dtype,
                )
                ref_neg = forward_stage1_pair_member(
                    self.brushnet_ref,
                    self.unet_ref,
                    noisy_neg.to(dtype=self.ref_forward_dtype),
                    timesteps_expanded,
                    encoder_hidden_states_expanded.to(dtype=self.ref_forward_dtype),
                    brushnet_cond.to(dtype=self.ref_forward_dtype),
                    self.ref_forward_dtype,
                )
            model_pos = forward_stage1_pair_member(
                self.brushnet, self.unet_main, noisy_pos, timesteps_expanded,
                encoder_hidden_states_expanded, brushnet_cond, self.forward_dtype,
            )
            model_neg = forward_stage1_pair_member(
                self.brushnet, self.unet_main, noisy_neg, timesteps_expanded,
                encoder_hidden_states_expanded, brushnet_cond, self.forward_dtype,
            )
            return torch.cat([model_pos, model_neg], dim=0), torch.cat([ref_pos, ref_neg], dim=0)

        noisy_all = torch.cat([noisy_pos, noisy_neg], dim=0)
        timesteps_all = timesteps_expanded.repeat(2)
        encoder_all = torch.cat([encoder_hidden_states_expanded, encoder_hidden_states_expanded], dim=0)
        cond_all = torch.cat([brushnet_cond, brushnet_cond], dim=0)
        with torch.no_grad():
            ref_pred = forward_stage1_pair_member(
                self.brushnet_ref, self.unet_ref, noisy_all.to(dtype=self.ref_forward_dtype), timesteps_all,
                encoder_all.to(dtype=self.ref_forward_dtype), cond_all.to(dtype=self.ref_forward_dtype),
                self.ref_forward_dtype,
            )
        model_pred = forward_stage1_pair_member(
            self.brushnet, self.unet_main, noisy_all, timesteps_all, encoder_all, cond_all, self.forward_dtype
        )
        return model_pred, ref_pred

    def _stage2_forward_pair(
        self,
        noisy_pos,
        noisy_neg,
        timesteps,
        timesteps_expanded,
        encoder_hidden_states,
        encoder_hidden_states_expanded,
        brushnet_cond,
    ):
        if self.split_pos_neg_forward:
            with torch.no_grad():
                ref_pos = forward_stage2_pair_member(
                    self.brushnet_ref,
                    self.unet_ref,
                    noisy_pos.to(dtype=self.ref_forward_dtype),
                    timesteps_expanded,
                    timesteps,
                    encoder_hidden_states_expanded.to(dtype=self.ref_forward_dtype),
                    encoder_hidden_states.to(dtype=self.ref_forward_dtype),
                    brushnet_cond.to(dtype=self.ref_forward_dtype),
                    self.ref_forward_dtype,
                    self.nframes,
                )
                ref_neg = forward_stage2_pair_member(
                    self.brushnet_ref,
                    self.unet_ref,
                    noisy_neg.to(dtype=self.ref_forward_dtype),
                    timesteps_expanded,
                    timesteps,
                    encoder_hidden_states_expanded.to(dtype=self.ref_forward_dtype),
                    encoder_hidden_states.to(dtype=self.ref_forward_dtype),
                    brushnet_cond.to(dtype=self.ref_forward_dtype),
                    self.ref_forward_dtype,
                    self.nframes,
                )
            model_pos = forward_stage2_pair_member(
                self.brushnet, self.unet_main, noisy_pos, timesteps_expanded, timesteps,
                encoder_hidden_states_expanded, encoder_hidden_states, brushnet_cond,
                self.forward_dtype, self.nframes,
            )
            model_neg = forward_stage2_pair_member(
                self.brushnet, self.unet_main, noisy_neg, timesteps_expanded, timesteps,
                encoder_hidden_states_expanded, encoder_hidden_states, brushnet_cond,
                self.forward_dtype, self.nframes,
            )
            return torch.cat([model_pos, model_neg], dim=0), torch.cat([ref_pos, ref_neg], dim=0)

        noisy_all = torch.cat([noisy_pos, noisy_neg], dim=0)
        timesteps_2d = timesteps_expanded.repeat(2)
        timesteps_motion = timesteps.repeat(2)
        encoder_2d = torch.cat([encoder_hidden_states_expanded, encoder_hidden_states_expanded], dim=0)
        encoder_motion = torch.cat([encoder_hidden_states, encoder_hidden_states], dim=0)
        cond_all = torch.cat([brushnet_cond, brushnet_cond], dim=0)
        with torch.no_grad():
            ref_pred = forward_stage2_pair_member(
                self.brushnet_ref,
                self.unet_ref,
                noisy_all.to(dtype=self.ref_forward_dtype),
                timesteps_2d,
                timesteps_motion,
                encoder_2d.to(dtype=self.ref_forward_dtype),
                encoder_motion.to(dtype=self.ref_forward_dtype),
                cond_all.to(dtype=self.ref_forward_dtype),
                self.ref_forward_dtype,
                self.nframes,
            )
        model_pred = forward_stage2_pair_member(
            self.brushnet,
            self.unet_main,
            noisy_all,
            timesteps_2d,
            timesteps_motion,
            encoder_2d,
            encoder_motion,
            cond_all,
            self.forward_dtype,
            self.nframes,
        )
        return model_pred, ref_pred

    def configure_optimizers(self):
        if self.use_8bit_adam:
            import bitsandbytes as bnb

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW
        params = [p for p in self.parameters() if p.requires_grad]
        return optimizer_class(
            params,
            lr=float(getattr(self, "learning_rate", self.base_learning_rate)),
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=self.adam_weight_decay,
            eps=self.adam_epsilon,
        )

    def on_before_optimizer_step(self, optimizer):
        params = [p for p in self.parameters() if p.requires_grad and p.grad is not None]
        if params:
            torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)

    def on_train_end(self):
        if not self.trainer.is_global_zero:
            return
        run_dir = Path(self.logdir or self.trainer.log_dir or ".")
        out_dir = run_dir / "last_weights"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.unet_main.save_pretrained(out_dir / "unet_main")
        self.brushnet.save_pretrained(out_dir / "brushnet")
        print(f"[official-diffueraser] last_weights={out_dir}", flush=True)

