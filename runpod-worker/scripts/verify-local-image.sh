#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${1:-truefaceswap-runpod-worker:local-test}"
PIPELINE_MODE="${2:-legacy}"

if [[ "${PIPELINE_MODE}" == "generation" ]]; then
  DOCKERFILE_ARGS=(-f "${ROOT_DIR}/Dockerfile.generation")
else
  DOCKERFILE_ARGS=()
fi

echo "[1/4] Building image: ${IMAGE_TAG} (${PIPELINE_MODE})"
docker build --progress=plain "${DOCKERFILE_ARGS[@]}" -t "${IMAGE_TAG}" "${ROOT_DIR}"

if [[ "${PIPELINE_MODE}" == "generation" ]]; then
  echo "[2/4] Verifying generation contract and example adapters"
  docker run --rm --entrypoint python3 "${IMAGE_TAG}" - <<'PY'
import importlib
import os
import shutil
import sys

required = ["requests", "runpod"]
for name in required:
    importlib.import_module(name)

sys.path.insert(0, "/worker/src")
import generation_contract  # noqa: F401

if not os.path.exists("/worker/scripts/example_generation_render.py"):
    raise RuntimeError("example_generation_render.py missing")
if not os.path.exists("/worker/scripts/example_generation_refine.py"):
    raise RuntimeError("example_generation_refine.py missing")
if shutil.which("ffmpeg") is None:
    raise RuntimeError("ffmpeg missing")
print("generation-import-check: ok")
PY

  echo "[3/4] Running generation adapter smoke test"
  docker run --rm --entrypoint bash "${IMAGE_TAG}" -lc '
set -euo pipefail
ffmpeg -y -f lavfi -i color=c=blue:s=720x1280 -frames:v 1 /tmp/source.jpg >/dev/null 2>&1
python3 - <<'"'"'PY'"'"'
from pathlib import Path
from generation_contract import CONTRACT_VERSION, IdentityImage, IdentityPack, RenderProfile, ShotPlan, save_identity_pack, save_shot_plan

pack = IdentityPack(
    version=CONTRACT_VERSION,
    primary_image="/tmp/source.jpg",
    images=[IdentityImage(path="/tmp/source.jpg", name="source.jpg", sha256="demo")],
)
save_identity_pack(Path("/tmp/identity_pack.json"), pack)

plan = ShotPlan(
    version=CONTRACT_VERSION,
    identity_pack_path="/tmp/identity_pack.json",
    prompt="cinematic portrait shot",
    negative_prompt=None,
    motion_preset="subtle_push_in",
    style_preset="cinematic",
    duration_seconds=2,
    seed=42,
    motion_reference_video=None,
    driving_audio=None,
    render_profile=RenderProfile(
        quality="balanced",
        aspect_ratio="9:16",
        fps=20,
        resolution=[720, 1280],
        frame_count=40,
    ),
)
save_shot_plan(Path("/tmp/shot_plan.json"), plan)
PY
python3 /worker/scripts/example_generation_render.py --shot-plan /tmp/shot_plan.json --output /tmp/rendered.mp4 --report /tmp/rendered.json
python3 /worker/scripts/example_generation_refine.py --identity-pack /tmp/identity_pack.json --input /tmp/rendered.mp4 --output /tmp/final.mp4 --report /tmp/refined.json
test -s /tmp/final.mp4
test -s /tmp/rendered.json
test -s /tmp/refined.json
'
else
  echo "[2/4] Verifying required Python modules and FaceFusion imports"
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
fi

echo "[4/4] Running worker preflight inside container"
docker run --rm -e WORKER_PIPELINE_MODE="${PIPELINE_MODE}" --entrypoint python3 "${IMAGE_TAG}" - <<'PY'
import sys

sys.path.insert(0, "/worker/src")
from preflight import run_preflight

print(run_preflight())
PY

echo "Local image verification passed for ${IMAGE_TAG}"
