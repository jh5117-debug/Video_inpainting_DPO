import unittest

from exp24_multibackbone_dpo_adapter.backends import (
    cococo,
    diffueraser,
    effecterase,
    floed,
    minimax_remover,
    propainter,
    vace,
    videocomposer,
    videopainter,
)


class BackendStatusTest(unittest.TestCase):
    def test_all_status_objects_import(self):
        modules = [
            diffueraser,
            videopainter,
            cococo,
            videocomposer,
            vace,
            minimax_remover,
            floed,
            effecterase,
            propainter,
        ]
        self.assertEqual(len(modules), 9)
        for module in modules:
            self.assertTrue(module.STATUS.model)

    def test_propainter_non_diffusion(self):
        self.assertEqual(propainter.STATUS.dpo_smoke_status, "NOT_APPLICABLE_NON_DIFFUSION_DPO")

    def test_effecterase_waiting_auth(self):
        self.assertIn("WAITING_AUTH", effecterase.STATUS.dpo_smoke_status)


if __name__ == "__main__":
    unittest.main()

