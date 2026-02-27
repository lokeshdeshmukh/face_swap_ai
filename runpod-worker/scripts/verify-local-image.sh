#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${1:-truefaceswap-runpod-worker:local-test}"

echo "[1/3] Building image: ${IMAGE_TAG}"
docker build --progress=plain -t "${IMAGE_TAG}" "${ROOT_DIR}"

echo "[2/3] Verifying required Python modules and FaceFusion imports"
docker run --rm --entrypoint python3 "${IMAGE_TAG}" - <<'PY'
import importlib
import shutil
import sys

required = ["requests", "runpod", "cv2", "onnxruntime", "scipy"]
for name in required:
    importlib.import_module(name)

if shutil.which("facefusion") is None:
    raise RuntimeError("facefusion executable not found")

sys.path.insert(0, "/opt/facefusion")
import facefusion.core  # noqa: F401
import facefusion.voice_extractor  # noqa: F401
import facefusion.processors.modules.face_swapper.core  # noqa: F401
print("module-import-check: ok")
PY

echo "[3/3] Running worker preflight inside container"
docker run --rm --entrypoint python3 "${IMAGE_TAG}" - <<'PY'
import sys

sys.path.insert(0, "/worker/src")
from preflight import run_preflight

print(run_preflight())
PY

echo "Local image verification passed for ${IMAGE_TAG}"
