import unittest

from exp23_two_stage_pool_morphology_sweep.code.exp23_trial_runner import (
    build_auto_eval_command,
    QUEUE_STATES,
)


class RunnerAutoEvalTest(unittest.TestCase):
    def test_auto_eval_uses_exp23_paired_davis50_script(self):
        cmd = build_auto_eval_command("unit_pair")
        self.assertEqual(cmd[:2], ["bash", "exp23_two_stage_pool_morphology_sweep/scripts/eval_exp23_pair001_davis50_pai.sh"])

    def test_append_only_queue_states_include_evaluating_and_completed(self):
        self.assertIn("evaluating", QUEUE_STATES)
        self.assertIn("completed", QUEUE_STATES)
        self.assertIn("failed", QUEUE_STATES)


if __name__ == "__main__":
    unittest.main()
