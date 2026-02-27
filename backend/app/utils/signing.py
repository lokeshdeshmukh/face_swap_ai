from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time


class TokenSigner:
    def __init__(self, secret: str) -> None:
        self.secret = secret.encode("utf-8")

    def sign(self, payload: dict[str, str | int]) -> str:
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        signature = hmac.new(self.secret, body, hashlib.sha256).hexdigest().encode("utf-8")
        return base64.urlsafe_b64encode(body + b"." + signature).decode("utf-8")

    def verify(self, token: str) -> dict[str, str | int]:
        try:
            decoded = base64.urlsafe_b64decode(token.encode("utf-8"))
            body, signature = decoded.rsplit(b".", 1)
            expected = hmac.new(self.secret, body, hashlib.sha256).hexdigest().encode("utf-8")
            if not hmac.compare_digest(signature, expected):
                raise ValueError("invalid token signature")
            payload = json.loads(body.decode("utf-8"))
            exp = int(payload.get("exp", 0))
            if exp < int(time.time()):
                raise ValueError("token expired")
            return payload
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, ValueError):
                raise
            raise ValueError("invalid token format") from exc


def sign_webhook(secret: str, body: bytes) -> str:
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def verify_webhook_signature(secret: str, body: bytes, signature: str) -> bool:
    expected = sign_webhook(secret, body)
    return hmac.compare_digest(expected, signature)
