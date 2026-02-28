from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import shutil
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


def _format_bytes(total_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(total_bytes)
    unit = units[0]
    for candidate in units:
        unit = candidate
        if value < 1024.0 or candidate == units[-1]:
            break
        value /= 1024.0
    return f"{value:.2f} {unit}"


def _dir_stats(path: Path, file_limit: int = 200000) -> tuple[int, int, bool]:
    file_count = 0
    total_bytes = 0
    truncated = False
    for root, _, files in os.walk(path):
        root_path = Path(root)
        for file_name in files:
            file_count += 1
            try:
                total_bytes += (root_path / file_name).stat().st_size
            except OSError:
                pass
            if file_count >= file_limit:
                truncated = True
                return file_count, total_bytes, truncated
    return file_count, total_bytes, truncated


def _log_cache_inventory(cache_root: Path) -> None:
    directories = {
        "xdg": cache_root / "xdg",
        "huggingface": cache_root / "huggingface",
        "torch": cache_root / "torch",
        "facefusion": cache_root / "facefusion",
    }
    logger.info("cache inventory root: %s", cache_root)
    for name, path in directories.items():
        if not path.exists():
            logger.info("cache inventory: %s missing", name)
            continue
        files, bytes_used, truncated = _dir_stats(path)
        suffix = " (truncated)" if truncated else ""
        logger.info(
            "cache inventory: %s files=%d size=%s%s",
            name,
            files,
            _format_bytes(bytes_used),
            suffix,
        )


def _configure_persistent_cache() -> None:
    volume_root = Path(os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume"))
    if not volume_root.exists() or not volume_root.is_dir():
        logger.info("persistent volume not available at %s; using ephemeral container storage", volume_root)
        return

    cache_root = volume_root / "truefaceswap-cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    xdg_cache = cache_root / "xdg"
    hf_home = cache_root / "huggingface"
    torch_home = cache_root / "torch"
    facefusion_home = cache_root / "facefusion"
    for path in (xdg_cache, hf_home, torch_home, facefusion_home):
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
    os.environ.setdefault("TORCH_HOME", str(torch_home))
    os.environ.setdefault("FACEFUSION_HOME", str(facefusion_home))

    root_facefusion_home = Path("/root/.facefusion")
    if root_facefusion_home.exists() and not root_facefusion_home.is_symlink():
        if root_facefusion_home.is_dir():
            try:
                shutil.copytree(root_facefusion_home, facefusion_home, dirs_exist_ok=True)
            except Exception:
                pass
            shutil.rmtree(root_facefusion_home, ignore_errors=True)
        else:
            root_facefusion_home.unlink(missing_ok=True)
    if not root_facefusion_home.exists():
        root_facefusion_home.symlink_to(facefusion_home, target_is_directory=True)

    logger.info("persistent cache enabled at %s", cache_root)
    _log_cache_inventory(cache_root)


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


def _extract_source_frames(source_video: Path, frames_dir: Path, max_frames: int, fps: float) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    pattern = frames_dir / "source_%03d.jpg"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_video),
        "-vf",
        f"fps={fps}",
        "-frames:v",
        str(max_frames),
        str(pattern),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f"ffmpeg source frame extraction failed: {result.stderr}")
    frames = sorted(frames_dir.glob("source_*.jpg"))
    if not frames:
        raise ValueError("no source frames could be extracted from source video")
    return frames


def _extract_first_frame(source_video: Path, out_path: Path) -> Path:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_video),
        "-frames:v",
        "1",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f"ffmpeg first-frame extraction failed: {result.stderr}")
    if not out_path.exists():
        raise ValueError("first frame extraction produced no output")
    return out_path


def _upload_file(url: str, path: Path, content_type: str) -> None:
    with path.open("rb") as f:
        response = requests.put(url, data=f, headers={"Content-Type": content_type}, timeout=600)
    response.raise_for_status()


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

    output_target = payload.get("output_target")
    output_upload_url: Optional[str] = None
    output_url: Optional[str] = None
    output_ref: Optional[str] = None
    if isinstance(output_target, dict):
        if output_target.get("upload_url"):
            output_upload_url = str(output_target["upload_url"])
        if output_target.get("output_url"):
            output_url = str(output_target["output_url"])
        if output_target.get("output_ref"):
            output_ref = str(output_target["output_ref"])

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
            source_video = tmp_path / "source_video.mp4"
            driving_audio = tmp_path / "driving_audio.wav"
            out_main = tmp_path / "result.mp4"
            source_images: list[Path] = []

            _download(str(assets["reference_video_url"]), reference_video)

            source_image_urls = assets.get("source_image_urls")
            if isinstance(source_image_urls, list):
                for idx, item in enumerate(source_image_urls):
                    if not item:
                        continue
                    image_path = tmp_path / f"source_{idx:03d}.jpg"
                    _download(str(item), image_path)
                    source_images.append(image_path)
            elif source_image_urls is not None:
                raise ValueError("source_image_urls must be a list of URLs")

            source_image_url = assets.get("source_image_url")
            if source_image_url and not source_images:
                legacy_source = tmp_path / "source_legacy.jpg"
                _download(str(source_image_url), legacy_source)
                source_images.append(legacy_source)

            source_video_url = assets.get("source_video_url")
            has_source_video = bool(source_video_url)
            if has_source_video:
                _download(str(source_video_url), source_video)
                max_frames = int(os.getenv("SOURCE_VIDEO_MAX_FRAMES", "24"))
                fps = float(os.getenv("SOURCE_VIDEO_SAMPLE_FPS", "2.0"))
                extracted = _extract_source_frames(source_video, tmp_path / "source_frames", max_frames=max_frames, fps=fps)
                source_images.extend(extracted)

            if mode == "video_swap" and not source_images:
                raise ValueError("video_swap requires source_image(s) or source_video")

            if mode == "video_swap":
                run_video_swap(
                    source_images=source_images,
                    target_video=reference_video,
                    output_video=out_main,
                    quality=quality,
                )
            elif mode == "photo_sing":
                photo_source_image: Path | None = source_images[0] if source_images else None
                if photo_source_image is None and has_source_video:
                    photo_source_image = _extract_first_frame(source_video, tmp_path / "source_from_video.jpg")
                if photo_source_image is None:
                    raise ValueError("photo_sing requires source_image(s) or source_video")
                if "driving_audio_url" in assets:
                    _download(str(assets["driving_audio_url"]), driving_audio)
                else:
                    _extract_audio(reference_video, driving_audio)
                run_photo_sing(
                    source_image=photo_source_image,
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

            success_body: Dict[str, Any] = {
                "job_id": job_id,
                "status": "completed",
                "metadata": {"mode": mode, "quality": quality},
            }
            if output_upload_url and (output_url or output_ref):
                _upload_file(output_upload_url, final_out, "video/mp4")
                if output_url:
                    success_body["output_url"] = output_url
                if output_ref:
                    success_body["output_ref"] = output_ref
            else:
                output_base64 = base64.b64encode(final_out.read_bytes()).decode("utf-8")
                success_body["output_base64"] = output_base64

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
    _configure_persistent_cache()
    try:
        preflight = run_preflight()
        logger.info("worker preflight passed: %s", preflight)
    except PreflightError as exc:
        logger.error("%s", exc)
        raise
    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
