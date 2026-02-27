from __future__ import annotations

import unittest

from app.utils.hash_utils import stable_config_hash


class TestConfigHash(unittest.TestCase):
    def test_stable(self) -> None:
        a = stable_config_hash(["video_swap", "balanced", "true"])
        b = stable_config_hash(["video_swap", "balanced", "true"])
        c = stable_config_hash(["video_swap", "fast", "true"])
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


if __name__ == "__main__":
    unittest.main()
