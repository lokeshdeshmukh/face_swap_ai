#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Worker wrapper for official LivePortrait inference")
    parser.add_argument("--source-image", required=True, dest="source_image")
    parser.add_argument("--driving-video", required=True, dest="driving_video")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")


def _weights_dir() -> Path:
    explicit = os.getenv("LIVEPORTRAIT_WEIGHTS_DIR", "").strip()
    if explicit:
        return Path(explicit)
    volume_root = Path(os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume"))
    if volume_root.exists():
        return volume_root / "truefaceswap-cache" / "liveportrait" / "pretrained_weights"
    return Path("/opt/liveportrait/pretrained_weights")


def _ensure_weights(repo_dir: Path, venv_bin_dir: Path) -> Path:
    weights_dir = _weights_dir()
    sentinel = weights_dir / "liveportrait" / "base_models" / "appearance_feature_extractor.pth"
    weights_dir.mkdir(parents=True, exist_ok=True)
    if not sentinel.exists():
        huggingface_cli = venv_bin_dir / "huggingface-cli"
        hf_repo = os.getenv("LIVEPORTRAIT_HF_REPO", "KlingTeam/LivePortrait").strip() or "KlingTeam/LivePortrait"
        _run(
            [
                str(huggingface_cli),
                "download",
                hf_repo,
                "--local-dir",
                str(weights_dir),
                "--exclude",
                "*.git*",
                "README.md",
                "docs",
            ]
        )

    target_link = repo_dir / "pretrained_weights"
    if target_link.is_symlink() or target_link.exists():
        if target_link.is_symlink() or target_link.is_file():
            target_link.unlink()
        else:
            shutil.rmtree(target_link)
    target_link.symlink_to(weights_dir, target_is_directory=True)
    return weights_dir


def _pick_output(animations_dir: Path, started_at: float) -> Path:
    candidates = [path for path in animations_dir.glob("*.mp4") if path.stat().st_mtime >= started_at - 1]
    if not candidates:
        raise SystemExit("liveportrait inference finished but no mp4 output was found")

    non_concat = [path for path in candidates if not path.stem.endswith("_concat")]
    pool = non_concat or candidates
    return max(pool, key=lambda path: path.stat().st_mtime)


def main() -> None:
    args = _parse_args()

    repo_dir = Path(os.getenv("LIVEPORTRAIT_REPO_DIR", "/opt/liveportrait"))
    venv_bin_dir = Path(os.getenv("LIVEPORTRAIT_VENV_BIN", "/opt/liveportrait-venv/bin"))
    liveportrait_python = venv_bin_dir / "python"
    inference_script = repo_dir / "inference.py"

    if not inference_script.exists():
        raise SystemExit(f"LivePortrait inference script not found: {inference_script}")
    if not liveportrait_python.exists():
        raise SystemExit(f"LivePortrait python runtime not found: {liveportrait_python}")

    _ensure_weights(repo_dir, venv_bin_dir)

    animations_dir = repo_dir / "animations"
    animations_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()

    command = [
        str(liveportrait_python),
        str(inference_script),
        "-s",
        str(Path(args.source_image).resolve()),
        "-d",
        str(Path(args.driving_video).resolve()),
        "--audio_priority",
        os.getenv("LIVEPORTRAIT_AUDIO_PRIORITY", "driving"),
        "--driving_option",
        os.getenv("LIVEPORTRAIT_DRIVING_OPTION", "expression-friendly"),
        "--animation_region",
        os.getenv("LIVEPORTRAIT_ANIMATION_REGION", "all"),
    ]
    if os.getenv("LIVEPORTRAIT_AUTO_CROP", "true").strip().lower() in {"1", "true", "yes", "on"}:
        command.append("--flag_crop_driving_video")

    _run(command, cwd=repo_dir)

    selected_output = _pick_output(animations_dir, started_at)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(selected_output, output_path)


if __name__ == "__main__":
    main()
