#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CogVideoX image-to-video generation")
    parser.add_argument("--shot-plan", required=True, dest="shot_plan")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    return parser.parse_args()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")


def _compose_prompt(base_prompt: str, motion_preset: str | None, style_preset: str | None) -> str:
    motion_map = {
        "cinematic_dolly": "cinematic dolly camera move, gentle forward motion",
        "subtle_push_in": "subtle push-in camera movement",
        "locked_portrait": "locked camera portrait framing, subtle natural micro-movements",
        "handheld_walk": "handheld camera movement with natural motion",
    }
    style_map = {
        "studio_realism": "studio lighting, realistic skin texture, high facial detail",
        "cinematic_realism": "cinematic lighting, realistic skin texture, premium film look",
        "editorial_beauty": "editorial beauty lighting, flattering facial detail",
    }

    parts = [base_prompt.strip()]
    if style_preset:
        parts.append(style_map.get(style_preset, style_preset.replace("_", " ")))
    if motion_preset:
        parts.append(motion_map.get(motion_preset, motion_preset.replace("_", " ")))
    parts.append("consistent facial identity, natural head motion, coherent temporal motion")
    return ", ".join(part for part in parts if part)


def _target_base_size(aspect_ratio: str) -> tuple[int, int]:
    # CogVideoX I2V is most reliable around 720x480-class resolutions.
    if aspect_ratio == "1:1":
        return (640, 640)
    if aspect_ratio == "4:5":
        return (608, 760)
    return (480, 720)


def _target_final_size(aspect_ratio: str, quality: str) -> tuple[int, int]:
    if aspect_ratio == "1:1":
        return (768, 768) if quality == "max" else (640, 640)
    if aspect_ratio == "4:5":
        return (864, 1080) if quality == "max" else (640, 800)
    return (1080, 1920) if quality == "max" else (720, 1280)


def main() -> None:
    args = _parse_args()

    import torch
    from diffusers import CogVideoXImageToVideoPipeline, CogVideoXDPMScheduler
    from diffusers.utils import export_to_video, load_image
    from PIL import Image, ImageOps

    from generation_contract import (
        CONTRACT_VERSION,
        AdapterReport,
        ensure_video_output,
        load_identity_pack,
        load_shot_plan,
        save_adapter_report,
    )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for CogVideoX generation but is not available")

    shot_plan = load_shot_plan(Path(args.shot_plan))
    identity_pack = load_identity_pack(Path(shot_plan.identity_pack_path))
    primary_image = identity_pack.primary_image
    if not primary_image:
        raise SystemExit("identity pack primary_image is required")

    model_id = os.getenv("GENERATION_MODEL_ID", "THUDM/CogVideoX-5b-I2V").strip()
    dtype_name = os.getenv("GENERATION_MODEL_DTYPE", "auto").strip().lower()
    if dtype_name == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype_name == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    offload_mode = os.getenv("GENERATION_OFFLOAD_MODE", "model").strip().lower()
    guidance_scale = _env_float("GENERATION_GUIDANCE_SCALE", 6.0)

    quality_defaults = {
        "fast": {"steps": 20, "frames": 33},
        "balanced": {"steps": 32, "frames": 49},
        "max": {"steps": 50, "frames": 65},
    }
    quality_cfg = quality_defaults.get(shot_plan.render_profile.quality, quality_defaults["balanced"])
    num_inference_steps = _env_int("GENERATION_NUM_INFERENCE_STEPS", quality_cfg["steps"])
    num_frames = _env_int("GENERATION_NUM_FRAMES", quality_cfg["frames"])
    fps = _env_int("GENERATION_OUTPUT_FPS", 8)

    prompt = _compose_prompt(shot_plan.prompt, shot_plan.motion_preset, shot_plan.style_preset)
    base_width, base_height = _target_base_size(shot_plan.render_profile.aspect_ratio)
    final_width, final_height = _target_final_size(shot_plan.render_profile.aspect_ratio, shot_plan.render_profile.quality)

    source_image = load_image(primary_image).convert("RGB")
    prepared_image = ImageOps.fit(source_image, (base_width, base_height), method=Image.Resampling.LANCZOS)

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    if offload_mode == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif offload_mode == "none":
        pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()

    seed = shot_plan.seed if shot_plan.seed is not None else 42
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.inference_mode():
        frames = pipe(
            prompt=prompt,
            image=prepared_image,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            generator=generator,
        ).frames[0]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="cogvideox-render-") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        raw_video = tmp_dir_path / "raw.mp4"
        final_video = tmp_dir_path / "final.mp4"
        export_to_video(frames, str(raw_video), fps=fps)

        vf = f"scale={final_width}:{final_height}:force_original_aspect_ratio=increase,crop={final_width}:{final_height}"
        cmd = ["ffmpeg", "-y", "-i", str(raw_video), "-vf", vf]
        if shot_plan.driving_audio:
            cmd.extend(["-i", shot_plan.driving_audio, "-map", "0:v:0", "-map", "1:a:0", "-shortest"])
            cmd.extend(["-c:a", "aac", "-b:a", "192k"])
        cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", str(final_video)])
        _run(cmd)
        final_video.replace(output_path)

    ensure_video_output(output_path)

    warnings: list[str] = []
    if len(identity_pack.images) > 1:
        warnings.append("Current CogVideoX backend uses the primary identity image only; extra identity images are not fused yet.")
    if identity_pack.identity_video:
        warnings.append("Identity video is not consumed by the first CogVideoX backend yet.")
    if shot_plan.motion_reference_video:
        warnings.append("Motion reference video is not consumed by the first CogVideoX backend yet.")
    if shot_plan.negative_prompt:
        warnings.append("Negative prompt is not consumed by the first CogVideoX backend yet.")

    if args.report:
        save_adapter_report(
            Path(args.report),
            AdapterReport(
                version=CONTRACT_VERSION,
                stage="generating",
                engine="cogvideox_i2v",
                model=model_id,
                metrics={
                    "num_inference_steps": num_inference_steps,
                    "num_frames": num_frames,
                    "fps": fps,
                    "seed": seed,
                    "width": final_width,
                    "height": final_height,
                },
                warnings=warnings,
            ),
        )


if __name__ == "__main__":
    main()
