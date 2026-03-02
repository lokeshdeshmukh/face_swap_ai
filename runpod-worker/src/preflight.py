from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List


class PreflightError(RuntimeError):
    pass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _check_python_modules(required: List[str]) -> List[str]:
    errors: List[str] = []
    for module_name in required:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - defensive in container
            errors.append(f"missing python module '{module_name}': {exc}")
    return errors


def _check_binaries(required: List[str]) -> List[str]:
    errors: List[str] = []
    for binary in required:
        if shutil.which(binary) is None:
            errors.append(f"missing executable '{binary}'")
    return errors


def _check_facefusion_processor_import() -> List[str]:
    try:
        if "/opt/facefusion" not in sys.path:
            sys.path.insert(0, "/opt/facefusion")
        importlib.import_module("facefusion.processors.modules.face_swapper.core")
        return []
    except Exception as exc:  # pragma: no cover - defensive in container
        return [f"cannot import facefusion face_swapper processor: {exc}"]


def _check_onnxruntime_cuda_provider() -> List[str]:
    try:
        ort = importlib.import_module("onnxruntime")
        providers = list(ort.get_available_providers())
    except Exception as exc:  # pragma: no cover - defensive in container
        return [f"cannot inspect onnxruntime providers: {exc}"]

    execution_provider = os.getenv("FACEFUSION_EXECUTION_PROVIDER", "cuda").strip().lower()
    if execution_provider == "cuda" and "CUDAExecutionProvider" not in providers:
        return [f"onnxruntime cuda provider not available (providers={providers})"]
    return []


def _check_torch_cuda() -> List[str]:
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - defensive in container
        return [f"cannot import torch: {exc}"]

    if not torch.cuda.is_available():
        return ["torch cuda is not available"]
    return []


def _check_liveportrait_runtime() -> List[str]:
    repo_dir = Path(os.getenv("LIVEPORTRAIT_REPO_DIR", "/opt/liveportrait"))
    venv_bin_dir = Path(os.getenv("LIVEPORTRAIT_VENV_BIN", "/opt/liveportrait-venv/bin"))

    errors: List[str] = []
    if not (repo_dir / "inference.py").exists():
        errors.append(f"liveportrait inference script not found at {repo_dir / 'inference.py'}")
    if not (venv_bin_dir / "python").exists():
        errors.append(f"liveportrait python runtime not found at {venv_bin_dir / 'python'}")
    if not any((venv_bin_dir / name).exists() for name in ("huggingface-cli", "hf")):
        errors.append(f"liveportrait Hugging Face CLI not found at {venv_bin_dir} (expected 'huggingface-cli' or 'hf')")
    return errors


def _check_mimicmotion_runtime() -> List[str]:
    repo_dir = Path(os.getenv("MIMICMOTION_REPO_DIR", "/opt/mimicmotion"))
    venv_bin_dir = Path(os.getenv("MIMICMOTION_VENV_BIN", "/opt/mimicmotion-venv/bin"))

    errors: List[str] = []
    if not (repo_dir / "inference.py").exists():
        errors.append(f"mimicmotion inference script not found at {repo_dir / 'inference.py'}")
    if not (venv_bin_dir / "python").exists():
        errors.append(f"mimicmotion python runtime not found at {venv_bin_dir / 'python'}")
    if not any((venv_bin_dir / name).exists() for name in ("huggingface-cli", "hf")):
        errors.append(f"mimicmotion Hugging Face CLI not found at {venv_bin_dir} (expected 'huggingface-cli' or 'hf')")
    return errors


def _portrait_backend_configured() -> bool:
    backend_name = os.getenv("PORTRAIT_REENACTMENT_BACKEND", "").strip().lower()
    backend_command = os.getenv("PORTRAIT_REENACTMENT_PIPELINE_COMMAND", "").strip()
    return bool(backend_command) or (backend_name not in {"", "unconfigured"})


def _full_body_backend_configured() -> bool:
    backend_name = os.getenv("FULL_BODY_REENACTMENT_BACKEND", "").strip().lower()
    backend_command = os.getenv("FULL_BODY_REENACTMENT_PIPELINE_COMMAND", "").strip()
    return bool(backend_command) or (backend_name not in {"", "unconfigured"})


def run_preflight() -> Dict[str, object]:
    errors: List[str] = []
    warnings: List[str] = []
    worker_pipeline_mode = os.getenv("WORKER_PIPELINE_MODE", "hybrid").strip().lower()

    if worker_pipeline_mode == "generation":
        errors.extend(
            _check_python_modules(["requests", "runpod", "torch", "diffusers", "transformers", "accelerate", "PIL", "cv2"])
        )
        errors.extend(_check_binaries(["ffmpeg"]))
        errors.extend(_check_torch_cuda())
        require_portrait_reenactment = _env_bool("REQUIRE_PORTRAIT_REENACTMENT_BACKEND", False)
        if require_portrait_reenactment or _portrait_backend_configured():
            portrait_reenactment_missing = _check_binaries(["liveportrait"])
            portrait_reenactment_runtime = _check_liveportrait_runtime()
            if portrait_reenactment_missing:
                if require_portrait_reenactment:
                    errors.extend(portrait_reenactment_missing)
                else:
                    warnings.extend(portrait_reenactment_missing)
            if portrait_reenactment_runtime:
                if require_portrait_reenactment:
                    errors.extend(portrait_reenactment_runtime)
                else:
                    warnings.extend(portrait_reenactment_runtime)
        require_full_body_reenactment = _env_bool("REQUIRE_FULL_BODY_REENACTMENT_BACKEND", False)
        if require_full_body_reenactment or _full_body_backend_configured():
            full_body_missing = _check_binaries(["mimicmotion"])
            full_body_runtime = _check_mimicmotion_runtime()
            if full_body_missing:
                if require_full_body_reenactment:
                    errors.extend(full_body_missing)
                else:
                    warnings.extend(full_body_missing)
            if full_body_runtime:
                if require_full_body_reenactment:
                    errors.extend(full_body_runtime)
                else:
                    warnings.extend(full_body_runtime)
    else:
        errors.extend(_check_python_modules(["requests", "runpod", "cv2", "onnxruntime", "scipy", "onnx"]))
        errors.extend(_check_binaries(["ffmpeg", "facefusion"]))
        errors.extend(_check_facefusion_processor_import())
        errors.extend(_check_onnxruntime_cuda_provider())

    require_photo_sing = _env_bool("REQUIRE_PHOTO_SING_DEPS", False)
    if require_photo_sing or worker_pipeline_mode != "generation":
        photo_sing_missing = _check_binaries(["liveportrait", "musetalk"])
        if photo_sing_missing:
            if require_photo_sing:
                errors.extend(photo_sing_missing)
            else:
                warnings.extend(photo_sing_missing)

    require_4k = _env_bool("REQUIRE_4K_ENHANCER", False)
    enhance_missing = _check_binaries(["realesrgan-ncnn-vulkan"])
    if enhance_missing:
        if require_4k:
            errors.extend(enhance_missing)
        else:
            warnings.extend(enhance_missing)

    if errors:
        joined = "\n - ".join(errors)
        raise PreflightError(f"worker preflight failed:\n - {joined}")

    return {"ok": True, "warnings": warnings}
