#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-body reenactment backend using MusePose")
    parser.add_argument("--shot-plan", required=True, dest="shot_plan")
    parser.add_argument("--identity-pack", required=True, dest="identity_pack")
    parser.add_argument("--control-bundle", required=True, dest="control_bundle")
    parser.add_argument("--source-image", required=True, dest="source_image")
    parser.add_argument("--driving-video", required=True, dest="driving_video")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
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


def _resolve_python_bin(repo_dir: Path) -> str:
    explicit = os.getenv("MUSEPOSE_PYTHON_BIN", "").strip()
    if explicit:
        return explicit
    candidate = Path(os.getenv("MUSEPOSE_VENV_BIN", "/opt/musepose-venv/bin")) / "python"
    if candidate.exists():
        return str(candidate)
    local_candidate = repo_dir / ".venv" / "bin" / "python"
    if local_candidate.exists():
        return str(local_candidate)
    return "python3"


def _resolve_repo_path(env_name: str, repo_dir: Path, default_relative: str) -> Path:
    explicit = os.getenv(env_name, "").strip()
    if explicit:
        return Path(explicit)
    return repo_dir / default_relative


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
            "Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) on the Runpod endpoint and make sure that token has access."
        ) from exc
    except (RepositoryNotFoundError, HfHubHTTPError) as exc:
        raise SystemExit(f"failed to download {filename} from {repo_id}: {exc}") from exc


def _download_snapshot(repo_id: str, local_dir: Path, cache_dir: Path, patterns: list[str] | None = None) -> Path:
    token = _hf_token()
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            cache_dir=str(cache_dir),
            token=token,
            allow_patterns=patterns,
        )
    except GatedRepoError as exc:
        raise SystemExit(
            f"gated Hugging Face repo access required for {repo_id}. "
            "Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) on the Runpod endpoint and make sure that token has access."
        ) from exc
    except (RepositoryNotFoundError, HfHubHTTPError) as exc:
        raise SystemExit(f"failed to download snapshot from {repo_id}: {exc}") from exc
    return local_dir


def _download_url(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return destination


def _weights_root() -> Path:
    explicit = os.getenv("MUSEPOSE_WEIGHTS_DIR", "").strip()
    if explicit:
        return Path(explicit)
    volume_root = Path(os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume"))
    if volume_root.exists():
        return volume_root / "truefaceswap-cache" / "musepose"
    return Path("/opt/musepose/models")


def _link_path(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.symlink_to(source, target_is_directory=source.is_dir())
    except OSError:
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def _ensure_weight_layout(repo_dir: Path) -> Path:
    weights_root = _weights_root()
    cache_dir = weights_root / ".hf-cache"
    pretrained_root = weights_root / "pretrained_weights"
    pretrained_root.mkdir(parents=True, exist_ok=True)

    musepose_dir = pretrained_root / "MusePose"
    dwpose_dir = pretrained_root / "DWPose"
    dwpose_dir_legacy = pretrained_root / "dwpose"
    sd_variations_dir = pretrained_root / "sd-image-variations-diffusers"
    sd_vae_dir = pretrained_root / "sd-vae-ft-mse"
    for path in (musepose_dir, dwpose_dir, sd_variations_dir, sd_vae_dir):
        path.mkdir(parents=True, exist_ok=True)

    musepose_repo = os.getenv("MUSEPOSE_HF_REPO", "TMElyralab/MusePose").strip() or "TMElyralab/MusePose"
    for filename in (
        "MusePose/denoising_unet.pth",
        "MusePose/motion_module.pth",
        "MusePose/pose_guider.pth",
        "MusePose/reference_unet.pth",
    ):
        downloaded = _download_file(musepose_repo, filename, cache_dir)
        _link_path(downloaded, musepose_dir / Path(filename).name)

    dwpose_repo = os.getenv("MUSEPOSE_DWPOSE_HF_REPO", "yzd-v/DWPose").strip() or "yzd-v/DWPose"
    dwpose_model = os.getenv("MUSEPOSE_DWPOSE_MODEL", "dw-ll_ucoco_384.pth").strip() or "dw-ll_ucoco_384.pth"
    downloaded = _download_file(dwpose_repo, dwpose_model, cache_dir)
    _link_path(downloaded, dwpose_dir / Path(dwpose_model).name)

    yolox_url = (
        os.getenv(
            "MUSEPOSE_YOLOX_URL",
            "https://download.openmmlab.com/mmdetection/v2.0/yolox/"
            "yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth",
        ).strip()
        or "https://download.openmmlab.com/mmdetection/v2.0/yolox/"
        "yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth"
    )
    yolox_name = os.getenv(
        "MUSEPOSE_YOLOX_FILENAME",
        "yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth",
    ).strip() or "yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth"
    yolox_path = _download_url(yolox_url, cache_dir / "external" / yolox_name)
    _link_path(yolox_path, dwpose_dir / yolox_name)
    _link_path(yolox_path, dwpose_dir / "yolox_l_8x8_300e_coco.pth")
    _link_path(dwpose_dir, dwpose_dir_legacy)

    sd_variations_repo = (
        os.getenv("MUSEPOSE_SD_VARIATIONS_REPO", "lambdalabs/sd-image-variations-diffusers").strip()
        or "lambdalabs/sd-image-variations-diffusers"
    )
    _download_snapshot(
        sd_variations_repo,
        sd_variations_dir,
        cache_dir,
        patterns=["unet/*", "image_encoder/*", "feature_extractor/*", "scheduler/*", "config.json", "model_index.json"],
    )
    for alias_name in ("image_encoder", "feature_extractor", "scheduler", "unet"):
        source_path = sd_variations_dir / alias_name
        if source_path.exists():
            _link_path(source_path, pretrained_root / alias_name)

    sd_vae_repo = os.getenv("MUSEPOSE_VAE_REPO", "stabilityai/sd-vae-ft-mse").strip() or "stabilityai/sd-vae-ft-mse"
    _download_snapshot(
        sd_vae_repo,
        sd_vae_dir,
        cache_dir,
        patterns=["*.json", "*.bin", "*.safetensors"],
    )

    repo_pretrained = repo_dir / "pretrained_weights"
    _link_path(pretrained_root, repo_pretrained)
    return pretrained_root


def _find_generated_video(output_dir: Path, exclude: set[Path]) -> Path:
    candidates = sorted(
        (
            path
            for path in output_dir.rglob("*.mp4")
            if path.is_file() and path.stat().st_size > 0 and path not in exclude
        ),
        key=lambda path: (path.stat().st_mtime, path.stat().st_size),
        reverse=True,
    )
    if not candidates:
        raise SystemExit(f"MusePose produced no mp4 output under {output_dir}")
    return candidates[0]


def _round_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return max(value, 1)
    rounded = int(round(value / multiple) * multiple)
    return max(rounded, multiple)


def _resolve_target_resolution(width: int, height: int) -> tuple[int, int]:
    explicit_width = os.getenv("MUSEPOSE_WIDTH", "").strip()
    explicit_height = os.getenv("MUSEPOSE_HEIGHT", "").strip()
    if explicit_width and explicit_height:
        return int(explicit_width), int(explicit_height)

    max_long_edge = int(os.getenv("MUSEPOSE_MAX_LONG_EDGE", "1024"))
    max_short_edge = int(os.getenv("MUSEPOSE_MAX_SHORT_EDGE", "576"))
    round_multiple = int(os.getenv("MUSEPOSE_RESOLUTION_MULTIPLE", "64"))

    long_edge = max(width, height)
    short_edge = min(width, height)
    scale = min(max_long_edge / max(long_edge, 1), max_short_edge / max(short_edge, 1), 1.0)

    scaled_width = _round_to_multiple(int(width * scale), round_multiple)
    scaled_height = _round_to_multiple(int(height * scale), round_multiple)
    return scaled_width, scaled_height


def _build_runtime_config(
    *,
    template_path: Path,
    output_dir: Path,
    source_image: Path,
    aligned_pose_video: Path,
    seed: int,
) -> Path:
    config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit(f"unexpected MusePose config format in {template_path}")
    config["seed"] = seed
    config["output_dir"] = str(output_dir)
    config["test_cases"] = {
        str(source_image.resolve()): [
            str(aligned_pose_video.resolve()),
        ]
    }
    runtime_path = output_dir / "musepose.runtime.yaml"
    runtime_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return runtime_path


def _validate_runtime_config(runtime_config: Path) -> None:
    config = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))
    test_cases = config.get("test_cases")
    if not isinstance(test_cases, dict) or not test_cases:
        raise SystemExit(f"invalid MusePose runtime config at {runtime_config}: missing test_cases")

    for ref_image_path, pose_paths in test_cases.items():
        ref_path = Path(str(ref_image_path))
        if not ref_path.exists():
            raise SystemExit(
                f"invalid MusePose runtime config at {runtime_config}: reference image path does not exist: {ref_image_path}"
            )
        if not isinstance(pose_paths, list) or not pose_paths:
            raise SystemExit(
                f"invalid MusePose runtime config at {runtime_config}: pose path list missing for {ref_image_path}"
            )
        for pose_path in pose_paths:
            candidate = Path(str(pose_path))
            if not candidate.exists():
                raise SystemExit(
                    f"invalid MusePose runtime config at {runtime_config}: pose video path does not exist: {pose_path}"
                )


def main() -> None:
    args = _parse_args()

    from generation_contract import ensure_video_output, load_control_bundle, load_shot_plan

    shot_plan = load_shot_plan(Path(args.shot_plan))
    control_bundle = load_control_bundle(Path(args.control_bundle))

    if shot_plan.task_type != "full_body_reenactment":
        raise SystemExit(
            f"full_body_reenactment backend expected task_type=full_body_reenactment, got {shot_plan.task_type}"
        )

    source_image = Path(args.source_image)
    driving_video = Path(args.driving_video)
    output_path = Path(args.output)
    if not source_image.exists():
        raise SystemExit(f"source image missing: {source_image}")
    if not driving_video.exists():
        raise SystemExit(f"driving video missing: {driving_video}")

    repo_dir = Path(os.getenv("MUSEPOSE_REPO_DIR", "/opt/musepose"))
    if not repo_dir.exists():
        raise SystemExit(
            f"MusePose repo not found at {repo_dir}. Install MusePose and set MUSEPOSE_REPO_DIR."
        )

    python_bin = _resolve_python_bin(repo_dir)
    pose_align_script = _resolve_repo_path("MUSEPOSE_ALIGN_SCRIPT", repo_dir, "pose_align.py")
    stage2_script = _resolve_repo_path("MUSEPOSE_STAGE2_SCRIPT", repo_dir, "test_stage_2.py")
    config_template = _resolve_repo_path("MUSEPOSE_CONFIG_PATH", repo_dir, "configs/test_stage_2.yaml")
    for required in (pose_align_script, stage2_script, config_template):
        if not required.exists():
            raise SystemExit(f"required MusePose file missing: {required}")

    width, height = shot_plan.render_profile.resolution
    width, height = _resolve_target_resolution(width, height)
    seed = shot_plan.seed if shot_plan.seed is not None else int(os.getenv("MUSEPOSE_SEED", "42"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = output_path.parent / f"{output_path.stem}.musepose"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_dir}:{pythonpath}" if pythonpath else str(repo_dir)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    _ensure_weight_layout(repo_dir)

    with tempfile.TemporaryDirectory(prefix="musepose-full-body-") as temp_dir:
        temp_root = Path(temp_dir)
        aligned_pose = temp_root / "pose-aligned.mp4"
        align_debug = temp_root / "pose-align-debug.mp4"

        align_cmd = [
            python_bin,
            str(pose_align_script),
            "--imgfn_refer",
            str(source_image.resolve()),
            "--vidfn",
            str(control_bundle.driving_video),
            "--outfn_align_pose_video",
            str(aligned_pose),
            "--outfn",
            str(align_debug),
        ]
        _run(align_cmd, cwd=repo_dir, env=env)
        if not aligned_pose.exists() or aligned_pose.stat().st_size <= 0:
            raise SystemExit(f"MusePose pose alignment produced no output video: {aligned_pose}")

        runtime_config = _build_runtime_config(
            template_path=config_template,
            output_dir=output_dir,
            source_image=source_image,
            aligned_pose_video=aligned_pose,
            seed=seed,
        )
        _validate_runtime_config(runtime_config)
        exclude = {aligned_pose.resolve()}
        stage2_cmd = [
            python_bin,
            str(stage2_script),
            "--config",
            str(runtime_config),
            "-W",
            str(width),
            "-H",
            str(height),
        ]
        _run(stage2_cmd, cwd=repo_dir, env=env)

    produced = _find_generated_video(output_dir, exclude=exclude)
    shutil.copyfile(produced, output_path)
    ensure_video_output(output_path)


if __name__ == "__main__":
    main()
