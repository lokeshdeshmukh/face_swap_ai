#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face refinement stage for full-body reenactment using GFPGAN")
    parser.add_argument("--shot-plan", required=True, dest="shot_plan")
    parser.add_argument("--identity-pack", required=True, dest="identity_pack")
    parser.add_argument("--control-bundle", required=True, dest="control_bundle")
    parser.add_argument("--source-image", required=True, dest="source_image")
    parser.add_argument("--driving-video", required=True, dest="driving_video")
    parser.add_argument("--input-video", required=True, dest="input_video")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        lines.append(line)
        print(line, flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\nstdout+stderr:\n" + "\n".join(lines))


def _weights_root() -> Path:
    explicit = os.getenv("GFPGAN_WEIGHTS_DIR", "").strip()
    if explicit:
        return Path(explicit)
    volume_root = Path(os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume"))
    if volume_root.exists():
        return volume_root / "truefaceswap-cache" / "gfpgan"
    return Path("/opt/gfpgan/models")


def _ensure_model_path() -> Path:
    from basicsr.utils.download_util import load_file_from_url

    weights_root = _weights_root()
    weights_root.mkdir(parents=True, exist_ok=True)
    default_model_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
    model_url = (
        os.getenv(
            "GFPGAN_MODEL_URL",
            default_model_url,
        ).strip()
        or default_model_url
    )
    model_name = Path(model_url).name or "GFPGANv1.3.pth"
    model_path = weights_root / model_name
    if not model_path.exists():
        downloaded = load_file_from_url(
            url=model_url,
            model_dir=str(weights_root),
            progress=True,
            file_name=model_name,
        )
        model_path = Path(downloaded)
    return model_path


def _probe_fps(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return 15.0
    rate = result.stdout.strip()
    if "/" in rate:
        num, den = rate.split("/", 1)
        try:
            den_v = float(den)
            return float(num) / den_v if den_v else 15.0
        except ValueError:
            return 15.0
    try:
        return float(rate)
    except ValueError:
        return 15.0


def _has_audio_stream(video_path: Path) -> bool:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and "audio" in result.stdout.lower()


def _restore_audio_if_missing(audio_source: Path, output_video: Path) -> None:
    if not audio_source.exists() or not output_video.exists():
        return
    if not _has_audio_stream(audio_source) or _has_audio_stream(output_video):
        return
    remuxed = output_video.with_name(f"{output_video.stem}.with-audio{output_video.suffix}")
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(output_video),
            "-i",
            str(audio_source),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(remuxed),
        ]
    )
    remuxed.replace(output_video)


def main() -> None:
    args = _parse_args()

    from generation_contract import ensure_video_output, load_control_bundle, load_shot_plan

    import cv2
    from gfpgan import GFPGANer

    input_video = Path(args.input_video)
    output_video = Path(args.output)
    if not input_video.exists():
        raise SystemExit(f"input video missing: {input_video}")
    output_video.parent.mkdir(parents=True, exist_ok=True)

    shot_plan = load_shot_plan(Path(args.shot_plan))
    control_bundle = load_control_bundle(Path(args.control_bundle))

    quality_key = shot_plan.render_profile.quality
    if quality_key == "max":
        blend = float(os.getenv("GFPGAN_BLEND_MAX", "1.0"))
        jpeg_quality = int(os.getenv("GFPGAN_FRAME_QUALITY_MAX", "98"))
    elif quality_key == "fast":
        blend = float(os.getenv("GFPGAN_BLEND_FAST", "0.65"))
        jpeg_quality = int(os.getenv("GFPGAN_FRAME_QUALITY_FAST", "94"))
    else:
        blend = float(os.getenv("GFPGAN_BLEND_BALANCED", "0.8"))
        jpeg_quality = int(os.getenv("GFPGAN_FRAME_QUALITY_BALANCED", "96"))

    model_path = _ensure_model_path()
    restorer = GFPGANer(
        model_path=str(model_path),
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )

    fps = max(_probe_fps(input_video), 1.0)

    with tempfile.TemporaryDirectory(prefix="gfpgan-face-refine-") as temp_dir:
        temp_root = Path(temp_dir)
        input_frames = temp_root / "input_frames"
        output_frames = temp_root / "output_frames"
        input_frames.mkdir(parents=True, exist_ok=True)
        output_frames.mkdir(parents=True, exist_ok=True)

        _run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_video),
                str(input_frames / "frame_%06d.png"),
            ]
        )

        for frame_path in sorted(input_frames.glob("frame_*.png")):
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            _, _, restored = restorer.enhance(
                frame,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=blend,
            )
            if restored is None:
                restored = frame
            cv2.imwrite(
                str(output_frames / frame_path.name),
                restored,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality] if frame_path.suffix.lower() in {".jpg", ".jpeg"} else [],
            )

        intermediate = temp_root / "refined.mp4"
        _run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                f"{fps:.6f}",
                "-i",
                str(output_frames / "frame_%06d.png"),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18" if quality_key == "max" else "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(intermediate),
            ]
        )

        shutil.copyfile(intermediate, output_video)

    driving_audio_source = Path(shot_plan.driving_audio) if shot_plan.driving_audio else Path(control_bundle.driving_video)
    _restore_audio_if_missing(driving_audio_source, output_video)
    ensure_video_output(output_video)


if __name__ == "__main__":
    main()
