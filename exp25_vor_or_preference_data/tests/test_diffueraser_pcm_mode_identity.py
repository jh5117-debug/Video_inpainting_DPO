import argparse
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "infer_diffueraser_or_exp25.py"
SMOKE = ROOT / "scripts" / "run_vor_or_model_smoke.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DiffuEraserPcmModeIdentityTest(unittest.TestCase):
    def sample_source(self) -> str:
        return '''class Demo:
    def __init__(self):
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            base_model_path,
            brushnet=self.brushnet
        ).to(self.device, torch.float16)
        ## use PCM
        self.ckpt = ckpt
        PCM_ckpts = checkpoints[ckpt][0].format(mode)
        self.guidance_scale = checkpoints[ckpt][2]
        if loaded != (ckpt + mode):
            # MODIFIED: use pcm_weights_path parameter instead of hardcoded path
            self.pipeline.load_lora_weights(
                pcm_weights_path, weight_name=PCM_ckpts, subfolder=mode
            )
            loaded = ckpt + mode

            if ckpt == "LCM-Like LoRA":
                self.pipeline.scheduler = LCMScheduler()
            else:
                self.pipeline.scheduler = TCDScheduler(
                    num_train_timesteps=1000,
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    timestep_spacing="trailing",
                )
        self.num_inference_steps = checkpoints[ckpt][1]
'''

    def test_official_pcm2_keeps_pcm_loader_and_disables_safety_checker(self):
        module = load_module(SCRIPT, "exp25_infer_pcm_test")
        patched = module.patch_diffueraser_or(self.sample_source(), "official_pcm2")
        self.assertIn("load_lora_weights", patched)
        self.assertIn("safety_checker=None", patched)
        self.assertIn("requires_safety_checker=False", patched)
        self.assertNotIn('self.ckpt = "none"', patched)

    def test_no_pcm_is_explicit_identity_without_pcm_loader(self):
        module = load_module(SCRIPT, "exp25_infer_no_pcm_test")
        patched = module.patch_diffueraser_or(self.sample_source(), "none")
        self.assertNotIn("load_lora_weights", patched)
        self.assertIn("Exp25 explicit no-PCM mode", patched)
        self.assertIn('self.ckpt = "none"', patched)
        self.assertIn("EXP25_NO_PCM_STEPS", patched)
        self.assertIn("safety_checker=None", patched)

    def test_generator_id_separates_pcm_modes(self):
        module = load_module(SMOKE, "exp25_smoke_generator_test")
        common = dict(
            model="diffueraser",
            prior_mode="propainter",
            no_pcm_steps=6,
            no_pcm_guidance=0.0,
            width=512,
            height=288,
            num_frames=24,
            diffueraser_path=Path("/weights/diffueraser"),
            propainter_model_dir=Path("/weights/propainter"),
        )
        no_pcm = argparse.Namespace(**common, pcm_mode="none")
        pcm = argparse.Namespace(**common, pcm_mode="official_pcm2")
        self.assertNotEqual(module.generator_id(no_pcm), module.generator_id(pcm))
        self.assertIn("none_propainter", module.generator_id(no_pcm))
        self.assertIn("official_pcm2_propainter", module.generator_id(pcm))


if __name__ == "__main__":
    unittest.main()
