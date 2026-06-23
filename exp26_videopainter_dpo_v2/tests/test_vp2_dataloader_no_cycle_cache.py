import inspect
import unittest

import exp26_videopainter_dpo_v2.code.train_videopainter_dpo_adapter as trainer


class TestVP2DataloaderNoCycleCache(unittest.TestCase):
    def test_no_itertools_cycle_import_or_usage(self):
        source = inspect.getsource(trainer)
        self.assertNotIn("itertools.cycle", source)
        self.assertIn("repeating_epoch_iterator", source)


if __name__ == "__main__":
    unittest.main()
