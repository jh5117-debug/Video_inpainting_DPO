import io
import tarfile
import unittest

from exp25_vor_or_preference_data.scripts.vor_archive_utils import sample_id_from_member, unsafe_member_reason


def info(name: str, *, linkname: str = "") -> tarfile.TarInfo:
    t = tarfile.TarInfo(name)
    if linkname:
        t.type = tarfile.SYMTYPE
        t.linkname = linkname
    else:
        t.size = 0
    return t


class TestVORArchiveSafety(unittest.TestCase):
    def test_reject_path_traversal(self):
        self.assertEqual(unsafe_member_reason(info("../bad")), "path_traversal")

    def test_reject_absolute_path(self):
        self.assertEqual(unsafe_member_reason(info("/abs/bad")), "absolute_path")

    def test_reject_unsafe_symlink(self):
        self.assertEqual(unsafe_member_reason(info("x/link", linkname="/etc/passwd")), "unsafe_link")

    def test_sample_id_heuristic(self):
        self.assertEqual(sample_id_from_member("VOR-Train/000123/frame_000.png"), "000123/frame_000.png")


if __name__ == "__main__":
    unittest.main()
