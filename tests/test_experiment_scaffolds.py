from __future__ import annotations

import importlib
import subprocess
import sys
import unittest


EXPERIMENT_MODULES = [
    "diffueraser_reproduction_sft",
    "official_videodpo_vc2",
    "official_videodpo_diffueraser",
    "official_videodpo_diffueraser_data_fullmask_loser",
    "official_videodpo_diffueraser_data_partialmask_loser_comp",
    "official_videodpo_diffueraser_data_partialmask_loser_nocomp",
    "official_videodpo_diffueraser_data_partialmask_loser_comp_k4",
    "official_videodpo_diffueraser_data_partialmask_loser_nocomp_k4",
    "official_videodpo_diffueraser_task_partialmask",
    "official_videodpo_diffueraser_youtubevos_partialmask_data",
    "official_videodpo_diffueraser_online_loser_generation",
]


class ExperimentScaffoldTests(unittest.TestCase):
    def test_experiment_modules_import(self) -> None:
        for module_name in EXPERIMENT_MODULES:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)

    def test_loser_generation_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "tools.offline_loser_generation", "--help"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertIn("--model_name", result.stdout)
        self.assertIn("--mask_mode", result.stdout)


if __name__ == "__main__":
    unittest.main()
