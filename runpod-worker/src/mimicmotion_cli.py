#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Worker wrapper for official MimicMotion inference")
    parser.add_argument("--source-image", required=True, dest="source_image")
    parser.add_argument("--driving-video", required=True, dest="driving_video")
    parser.add_argument("--output", required=True)
    parser.add_argument("--frame-count", type=int, default=25)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--resolution", type=int, default=576)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result.stdout


def _hf_token() -> str | None:
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def _download_file(repo_id: str, filename: str, cache_dir: Path) -> Path:
    token = _hf_token()
    try:
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(cache_dir),
                token=token,
            )
        )
    except GatedRepoError as exc:
        raise SystemExit(
            f"gated Hugging Face repo access required for {repo_id}. "
            "Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) on the Runpod endpoint and make sure that token has accepted access to the model."
        ) from exc
    except (RepositoryNotFoundError, HfHubHTTPError) as exc:
        raise SystemExit(f"failed to download {filename} from {repo_id}: {exc}") from exc


def _download_snapshot(repo_id: str, local_dir: Path, cache_dir: Path) -> Path:
    token = _hf_token()
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            cache_dir=str(cache_dir),
            token=token,
        )
    except GatedRepoError as exc:
        raise SystemExit(
            f"gated Hugging Face repo access required for {repo_id}. "
            "Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) on the Runpod endpoint and make sure that token has accepted access to the model."
        ) from exc
    except (RepositoryNotFoundError, HfHubHTTPError) as exc:
        raise SystemExit(f"failed to download snapshot from {repo_id}: {exc}") from exc
    return local_dir


def _weights_root() -> Path:
    explicit = os.getenv("MIMICMOTION_WEIGHTS_DIR", "").strip()
    if explicit:
        return Path(explicit)
    volume_root = Path(os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume"))
    if volume_root.exists():
        return volume_root / "truefaceswap-cache" / "mimicmotion"
    return Path("/opt/mimicmotion/models")


def _ensure_repo_path(repo_dir: Path, relative_path: str) -> Path:
    candidate = repo_dir / relative_path
    if not candidate.exists():
        raise SystemExit(f"required MimicMotion file missing: {candidate}")
    return candidate


def _ensure_weights(repo_dir: Path, venv_bin_dir: Path) -> tuple[Path, Path]:
    weights_root = _weights_root()
    checkpoints_dir = weights_root / "checkpoints"
    dwpose_dir = weights_root / "DWPose"
    base_model_dir = weights_root / "base_model"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    dwpose_dir.mkdir(parents=True, exist_ok=True)
    base_model_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name = os.getenv("MIMICMOTION_CKPT_NAME", "MimicMotion_1-1.pth").strip() or "MimicMotion_1-1.pth"
    checkpoint_path = checkpoints_dir / ckpt_name
    base_model_repo = (
        os.getenv("MIMICMOTION_BASE_MODEL", "stabilityai/stable-video-diffusion-img2vid-xt-1-1").strip()
        or "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
    )
    dwpose_repo = os.getenv("MIMICMOTION_DWPOSE_HF_REPO", "yzd-v/DWPose").strip() or "yzd-v/DWPose"
    dwpose_files = [
        ("yolox_l.onnx", dwpose_dir / "yolox_l.onnx"),
        ("dw-ll_ucoco_384.onnx", dwpose_dir / "dw-ll_ucoco_384.onnx"),
    ]
    cache_dir = weights_root / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.exists():
        hf_repo = os.getenv("MIMICMOTION_HF_REPO", "Tencent/MimicMotion").strip() or "Tencent/MimicMotion"
        downloaded_path = _download_file(hf_repo, ckpt_name, cache_dir)
        shutil.copyfile(downloaded_path, checkpoint_path)
    for filename, target_path in dwpose_files:
        if target_path.exists():
            continue
        downloaded_path = _download_file(dwpose_repo, filename, cache_dir)
        shutil.copyfile(downloaded_path, target_path)
    if not (base_model_dir / "unet" / "config.json").exists():
        _download_snapshot(base_model_repo, base_model_dir, cache_dir)

    models_dir = repo_dir / "models"
    if weights_root != models_dir:
        if models_dir.is_symlink() or models_dir.exists():
            if models_dir.is_symlink() or models_dir.is_file():
                models_dir.unlink()
            else:
                shutil.rmtree(models_dir)
        models_dir.symlink_to(weights_root, target_is_directory=True)
    return checkpoint_path, base_model_dir


def _ffprobe_frame_count(video_path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return 0
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def _detect_config_flag(inference_script: Path, python_bin: Path, repo_dir: Path, env: dict[str, str]) -> str | None:
    result = subprocess.run(
        [str(python_bin), str(inference_script), "--help"],
        cwd=str(repo_dir),
        env=env,
        capture_output=True,
        text=True,
    )
    help_text = f"{result.stdout}\n{result.stderr}"
    for flag in ("--config", "--inference_config", "--inference-config"):
        if flag in help_text:
            return flag
    return None


def _build_config_text(
    source_image: Path,
    driving_video: Path,
    checkpoint_path: Path,
    base_model_path: Path,
    frame_count: int,
    fps: int,
    resolution: int,
    seed: int,
) -> str:
    sample_stride = int(os.getenv("MIMICMOTION_SAMPLE_STRIDE", "2"))
    overlap = int(os.getenv("MIMICMOTION_FRAMES_OVERLAP", "6"))
    steps = int(os.getenv("MIMICMOTION_NUM_INFERENCE_STEPS", "25"))
    guidance = float(os.getenv("MIMICMOTION_GUIDANCE_SCALE", "2.2"))
    noise = float(os.getenv("MIMICMOTION_NOISE_AUG_STRENGTH", "0.0"))

    return "\n".join(
        [
            f'base_model_path: "{base_model_path.resolve()}"',
            f'ckpt_path: "{checkpoint_path}"',
            "test_case:",
            f'  - ref_video_path: "{driving_video.resolve()}"',
            f'    ref_image_path: "{source_image.resolve()}"',
            f"    num_frames: {frame_count}",
            f"    resolution: {resolution}",
            f"    frames_overlap: {overlap}",
            f"    num_inference_steps: {steps}",
            f"    noise_aug_strength: {noise}",
            f"    guidance_scale: {guidance}",
            f"    sample_stride: {sample_stride}",
            f"    fps: {fps}",
            f"    seed: {seed}",
            "",
        ]
    )


def _pick_output(output_roots: list[Path], started_at: float) -> Path:
    candidates: list[Path] = []
    for root in output_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.mp4"):
            try:
                if path.stat().st_mtime >= started_at - 1:
                    candidates.append(path)
            except FileNotFoundError:
                continue

    if not candidates:
        raise SystemExit("mimicmotion inference finished but no mp4 output was found")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> None:
    args = _parse_args()

    repo_dir = Path(os.getenv("MIMICMOTION_REPO_DIR", "/opt/mimicmotion"))
    venv_bin_dir = Path(os.getenv("MIMICMOTION_VENV_BIN", "/opt/mimicmotion-venv/bin"))
    python_bin = venv_bin_dir / "python"
    inference_script = _ensure_repo_path(repo_dir, "inference.py")
    if not python_bin.exists():
        raise SystemExit(f"MimicMotion python runtime not found: {python_bin}")

    checkpoint_path, base_model_path = _ensure_weights(repo_dir, venv_bin_dir)
    requested_frame_count = max(1, args.frame_count)
    probed_frame_count = _ffprobe_frame_count(Path(args.driving_video))
    frame_count = min(requested_frame_count, probed_frame_count) if probed_frame_count else requested_frame_count
    resolution = max(256, args.resolution)

    with tempfile.TemporaryDirectory(prefix="mimicmotion-config-") as temp_dir:
        temp_path = Path(temp_dir)
        config_path = temp_path / "test.yaml"
        config_path.write_text(
            _build_config_text(
                source_image=Path(args.source_image),
                driving_video=Path(args.driving_video),
                checkpoint_path=checkpoint_path,
                base_model_path=base_model_path,
                frame_count=frame_count,
                fps=max(1, args.fps),
                resolution=resolution,
                seed=args.seed,
            ),
            encoding="utf-8",
        )

        output_dir = repo_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        started_at = time.time()
        env = os.environ.copy()
        cache_root = _weights_root()
        env.setdefault("HF_HOME", str(cache_root / "hf_home"))
        env.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "hf_cache"))

        config_flag = _detect_config_flag(inference_script, python_bin, repo_dir, env)
        command = [str(python_bin), str(inference_script)]
        if config_flag:
            command.extend([config_flag, str(config_path)])
        else:
            command.append(str(config_path))

        _run(command, cwd=repo_dir, env=env)

        selected_output = _pick_output(
            [
                output_dir,
                repo_dir / "results",
                repo_dir / "result",
                repo_dir / "animations",
            ],
            started_at,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(selected_output, output_path)


if __name__ == "__main__":
    main()
