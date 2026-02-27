from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import runpod

from preflight import PreflightError, run_preflight
from pipeline import PipelineError, run_4k_enhance, run_photo_sing, run_video_swap

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("runpod-worker")


def _download(url: str, path: Path) -> None:
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    path.write_bytes(response.content)


def _sign(secret: str, payload: bytes) -> str:
    return hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def _callback(url: str, secret: str, body: Dict[str, Any]) -> None:
    raw = json.dumps(body, separators=(",", ":")).encode("utf-8")
    signature = _sign(secret, raw)
    headers = {
        "Content-Type": "application/json",
        "X-Callback-Signature": signature,
    }
    response = requests.post(url, data=raw, headers=headers, timeout=180)
    response.raise_for_status()


def _extract_audio(video_path: Path, audio_out: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(audio_out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f"ffmpeg audio extraction failed: {result.stderr}")


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    payload = event.get("input", {}) if isinstance(event, dict) else {}
    if not isinstance(payload, dict):
        raise ValueError("invalid event payload")

    job_id = str(payload["job_id"])
    mode = str(payload["mode"])
    quality = str(payload.get("quality", "balanced"))
    raw_enable_4k = payload.get("enable_4k", False)
    if isinstance(raw_enable_4k, str):
        enable_4k = raw_enable_4k.lower() in {"1", "true", "yes", "on"}
    else:
        enable_4k = bool(raw_enable_4k)

    assets = payload.get("assets")
    if not isinstance(assets, dict):
        raise ValueError("assets field is required")

    callback = payload.get("callback")
    callback_url: Optional[str] = None
    callback_secret: Optional[str] = None
    if isinstance(callback, dict) and callback.get("url") and callback.get("secret"):
        callback_url = str(callback["url"])
        callback_secret = str(callback["secret"])

    try:
        with tempfile.TemporaryDirectory(prefix="truefaceswap-") as tmp:
            tmp_path = Path(tmp)
            reference_video = tmp_path / "reference.mp4"
            source_image = tmp_path / "source.jpg"
            driving_audio = tmp_path / "driving_audio.wav"
            out_main = tmp_path / "result.mp4"

            _download(str(assets["reference_video_url"]), reference_video)
            _download(str(assets["source_image_url"]), source_image)

            if mode == "video_swap":
                run_video_swap(source_image=source_image, target_video=reference_video, output_video=out_main, quality=quality)
            elif mode == "photo_sing":
                if "driving_audio_url" in assets:
                    _download(str(assets["driving_audio_url"]), driving_audio)
                else:
                    _extract_audio(reference_video, driving_audio)
                run_photo_sing(
                    source_image=source_image,
                    driving_video=reference_video,
                    driving_audio=driving_audio,
                    output_video=out_main,
                )
            else:
                raise ValueError(f"unsupported mode: {mode}")

            final_out = out_main
            if enable_4k:
                enhanced = tmp_path / "result_4k.mp4"
                run_4k_enhance(out_main, enhanced)
                final_out = enhanced

            output_base64 = base64.b64encode(final_out.read_bytes()).decode("utf-8")
            success_body = {
                "job_id": job_id,
                "status": "completed",
                "output_base64": output_base64,
                "metadata": {"mode": mode, "quality": quality},
            }
            if callback_url and callback_secret:
                try:
                    _callback(callback_url, callback_secret, success_body)
                except requests.RequestException:
                    pass
            return {"ok": True, "job_id": job_id, "status": "completed", **success_body}

    except (requests.RequestException, PipelineError, ValueError, KeyError) as exc:
        failure_body = {
            "job_id": job_id,
            "status": "failed",
            "error": str(exc),
        }
        if callback_url and callback_secret:
            try:
                _callback(callback_url, callback_secret, failure_body)
            except Exception:
                pass
        return {"ok": False, "job_id": job_id, "error": str(exc)}


def main() -> None:
    logger.info("worker boot: python=%s", sys.version.replace("\n", " "))
    logger.info(
        "worker env: endpoint_id=%s has_webhook=%s",
        os.getenv("RUNPOD_ENDPOINT_ID", ""),
        bool(os.getenv("RUNPOD_WEBHOOK_GET_JOB")),
    )
    try:
        preflight = run_preflight()
        logger.info("worker preflight passed: %s", preflight)
    except PreflightError as exc:
        logger.error("%s", exc)
        raise
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
