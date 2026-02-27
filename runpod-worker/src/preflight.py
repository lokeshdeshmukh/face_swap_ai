from __future__ import annotations

import importlib
import os
import shutil
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


def run_preflight() -> Dict[str, object]:
    errors: List[str] = []
    warnings: List[str] = []

    errors.extend(_check_python_modules(["requests", "runpod", "cv2", "onnxruntime"]))
    errors.extend(_check_binaries(["ffmpeg", "facefusion"]))

    require_photo_sing = _env_bool("REQUIRE_PHOTO_SING_DEPS", False)
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
