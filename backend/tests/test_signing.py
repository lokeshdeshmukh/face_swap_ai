from __future__ import annotations

import time
import unittest

from app.utils.signing import TokenSigner


class TestTokenSigner(unittest.TestCase):
    def test_round_trip(self) -> None:
        signer = TokenSigner("secret")
        token = signer.sign({"path": "/tmp/a.mp4", "exp": int(time.time()) + 60})
        payload = signer.verify(token)
        self.assertEqual(payload["path"], "/tmp/a.mp4")

    def test_expired(self) -> None:
        signer = TokenSigner("secret")
        token = signer.sign({"path": "/tmp/a.mp4", "exp": int(time.time()) - 1})
        with self.assertRaises(ValueError):
            signer.verify(token)

    def test_invalid_format(self) -> None:
        signer = TokenSigner("secret")
        with self.assertRaises(ValueError):
            signer.verify("not-a-valid-token")


if __name__ == "__main__":
    unittest.main()
