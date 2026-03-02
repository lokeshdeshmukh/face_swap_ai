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


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def _env_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return int(raw)


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


def _compose_motion_reference_phrase(profile: dict[str, object] | None) -> str | None:
    if not profile:
        return None
    motion_type = str(profile.get("motion_type") or "").strip()
    motion_summary = str(profile.get("motion_summary") or "").strip()
    mapping = {
        "push_in": "camera movement should follow a gentle push-in from the reference clip",
        "pull_out": "camera movement should follow a gentle pull-back from the reference clip",
        "pan_left": "camera movement should follow a leftward pan from the reference clip",
        "pan_right": "camera movement should follow a rightward pan from the reference clip",
        "tilt_up": "camera movement should follow an upward tilt from the reference clip",
        "tilt_down": "camera movement should follow a downward tilt from the reference clip",
        "handheld": "camera movement should mimic natural handheld motion from the reference clip",
        "mixed_motion": "camera movement should follow the mixed motion pattern from the reference clip",
        "locked_portrait": "camera framing should stay mostly locked like the reference clip",
    }
    if motion_type in mapping:
        return mapping[motion_type]
    if motion_summary:
        return f"camera movement should follow this reference behavior: {motion_summary}"
    return None


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


def _prepare_single_image(source_image, size: tuple[int, int], image_ops, image_module):
    return image_ops.fit(source_image, size, method=image_module.Resampling.LANCZOS)


def _score_identity_image(image, image_filter_module, image_stat_module) -> float:
    grayscale = image.convert("L")
    stat = image_stat_module.Stat(grayscale)
    contrast = (stat.stddev[0] / 64.0) if stat.stddev else 0.0

    edges = grayscale.filter(image_filter_module.FIND_EDGES)
    edge_stat = image_stat_module.Stat(edges)
    edge_mean = (edge_stat.mean[0] / 255.0) if edge_stat.mean else 0.0

    width, height = image.size
    megapixels = min((width * height) / 1_000_000.0, 12.0)
    aspect_ratio = width / max(height, 1)
    portrait_target = 0.75
    aspect_score = max(0.0, 1.0 - abs(aspect_ratio - portrait_target))

    return (megapixels * 0.35) + (contrast * 1.5) + (edge_mean * 4.0) + aspect_score


def _select_primary_index(loaded_images: list[dict[str, object]]) -> tuple[int, str]:
    manual_index = _env_optional_int("GENERATION_HERO_IMAGE_INDEX")
    if manual_index is not None:
        bounded = max(0, min(manual_index, len(loaded_images) - 1))
        return bounded, "manual"

    ranked = sorted(
        enumerate(loaded_images),
        key=lambda item: float(item[1]["score"]),
        reverse=True,
    )
    return ranked[0][0], "auto"


def _prepare_identity_reference_image(identity_pack, size: tuple[int, int], image_ops, image_module):
    from PIL import ImageFilter, ImageStat

    paths = [image.path for image in identity_pack.images if image.path]
    if not paths:
        raise SystemExit("identity pack contains no images")

    multi_image_mode = _env_str("GENERATION_MULTI_IMAGE_MODE", "hero_grid").lower()
    loaded_images: list[dict[str, object]] = []
    for path in paths[:4]:
        image = image_module.open(path).convert("RGB")
        loaded_images.append(
            {
                "path": path,
                "image": image,
                "score": _score_identity_image(image, ImageFilter, ImageStat),
            }
        )

    hero_index, hero_selection = _select_primary_index(loaded_images)

    if len(loaded_images) == 1 or multi_image_mode == "primary_only":
        hero_image = loaded_images[hero_index if hero_index < len(loaded_images) else 0]["image"]
        return _prepare_single_image(hero_image, size, image_ops, image_module), {
            "multi_image_mode": "primary_only",
            "identity_images_used": 1,
            "hero_index": hero_index,
            "hero_selection": hero_selection,
        }

    width, height = size
    canvas = image_module.new("RGB", (width, height), (10, 12, 18))
    pad = max(8, width // 48)
    side_width = max(width // 3, 160)
    hero_width = width - side_width - (pad * 3)
    hero_height = height - (pad * 2)
    tile_height = max((hero_height - (pad * 2)) // 3, 96)

    hero_image = image_ops.fit(
        loaded_images[hero_index if hero_index < len(loaded_images) else 0]["image"],
        (hero_width, hero_height),
        method=image_module.Resampling.LANCZOS,
    )
    canvas.paste(hero_image, (pad, pad))

    accent_colors = [(42, 96, 255), (26, 188, 156), (244, 208, 63)]
    support_index = 0
    y_cursor = pad
    for idx, loaded in enumerate(loaded_images):
        if idx == hero_index:
            continue
        if support_index >= 3:
            break
        tile = image_ops.fit(
            loaded["image"],
            (side_width - (pad * 2), tile_height),
            method=image_module.Resampling.LANCZOS,
        )
        tile_x = hero_width + (pad * 2)
        tile_y = y_cursor
        border_color = accent_colors[support_index % len(accent_colors)]
        border = image_module.new("RGB", (tile.width + pad, tile.height + pad), border_color)
        border.paste(tile, (pad // 2, pad // 2))
        canvas.paste(border, (tile_x, tile_y))
        y_cursor += border.height + pad
        support_index += 1

    return canvas, {
        "multi_image_mode": "hero_grid",
        "identity_images_used": min(len(loaded_images), 4),
        "hero_index": hero_index,
        "hero_selection": hero_selection,
    }


def main() -> None:
    args = _parse_args()

    import torch
    from diffusers import CogVideoXImageToVideoPipeline, CogVideoXDPMScheduler
    from diffusers.utils import export_to_video
    from PIL import Image, ImageFilter, ImageOps, ImageStat

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
    motion_reference_phrase = _compose_motion_reference_phrase(shot_plan.motion_reference_profile)
    if motion_reference_phrase:
        prompt = f"{prompt}, {motion_reference_phrase}"
    base_width, base_height = _target_base_size(shot_plan.render_profile.aspect_ratio)
    final_width, final_height = _target_final_size(shot_plan.render_profile.aspect_ratio, shot_plan.render_profile.quality)

    prepared_image, image_prep_metadata = _prepare_identity_reference_image(
        identity_pack,
        (base_width, base_height),
        ImageOps,
        Image,
    )

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
    if len(identity_pack.images) > 1 and image_prep_metadata.get("multi_image_mode") == "hero_grid":
        warnings.append("Multiple identity images were packed into a reference canvas; learned identity fusion is not implemented yet.")
    elif len(identity_pack.images) > 1:
        warnings.append("Multiple identity images were provided, but only one image was used because GENERATION_MULTI_IMAGE_MODE=primary_only.")
    if identity_pack.identity_video:
        warnings.append("Identity video is not consumed by the first CogVideoX backend yet.")
    if shot_plan.motion_reference_profile:
        warnings.append("Motion reference video is consumed through derived motion-profile prompting, not direct control-video conditioning.")
    elif shot_plan.motion_reference_video:
        warnings.append("Motion reference video was provided but motion analysis did not resolve a usable profile.")
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
                    **image_prep_metadata,
                    "motion_profile_used": "yes" if shot_plan.motion_reference_profile else "no",
                },
                warnings=warnings,
            ),
        )


if __name__ == "__main__":
    main()
