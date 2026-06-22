import unittest
from pathlib import Path
from unittest import mock

from exp25_vor_or_preference_data.scripts.transfer_effecterase_hal_to_pai import remote_finalize


class AtomicRemoteFinalizeTest(unittest.TestCase):
    def test_uses_mv_and_sync(self):
        with mock.patch("subprocess.run") as run:
            remote_finalize(Path("/k"), "host", "/tmp/a.partial", "/tmp/a")
            cmd = run.call_args.args[0]
            self.assertIn("ssh", cmd[0])
            joined = " ".join(cmd)
            self.assertIn("mv -f", joined)
            self.assertIn("sync", joined)


if __name__ == "__main__":
    unittest.main()

