FROM runpod/base:0.4.1-cuda12.1.0

WORKDIR /worker

ARG FACEFUSION_REF=3.5.3
ARG ONNXRUNTIME_VERSION=1.23.2
ARG SCIPY_PY310_VERSION=1.15.3

RUN apt-get update && \
    apt-get install -y --no-install-recommends git ffmpeg libcudnn9-cuda-12 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /worker/requirements.txt
RUN set -eux; \
    ln -sf "$(command -v python3)" /usr/local/bin/python; \
    python3 -m ensurepip --upgrade || true; \
    python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel; \
    python3 -m pip install --no-cache-dir -r /worker/requirements.txt; \
    python3 -c "import requests, runpod; print('deps-ok')"

RUN git clone --depth 1 --branch "${FACEFUSION_REF}" https://github.com/facefusion/facefusion.git /opt/facefusion

RUN set -eux; \
    find /opt/facefusion -type f -name "requirements*.txt" -print0 | xargs -0 sed -i "s/scipy==1.17.0/scipy==${SCIPY_PY310_VERSION}/g"; \
    cd /opt/facefusion; \
    (python3 install.py --skip-conda --onnxruntime cuda || python3 install.py --skip-conda --onnxruntime default)

RUN set -eux; \
    python3 -m pip install --no-cache-dir "scipy==${SCIPY_PY310_VERSION}"; \
    python3 -m pip install --no-cache-dir "onnx==1.20.1"; \
    python3 -m pip install --no-cache-dir opencv-python-headless==4.13.0.92; \
    (python3 -m pip install --no-cache-dir "onnxruntime-gpu==${ONNXRUNTIME_VERSION}" || python3 -m pip install --no-cache-dir "onnxruntime==${ONNXRUNTIME_VERSION}")

RUN set -eux; \
    python3 -c "import cv2, onnxruntime, scipy, onnx; print('facefusion-runtime-ok')"; \
    python3 -c "import onnxruntime as ort; p=ort.get_available_providers(); print('onnxruntime providers:', p); assert 'CUDAExecutionProvider' in p, p"; \
    cd /opt/facefusion; \
    python3 -c "import facefusion.processors.modules.face_swapper.core as _m; print('face_swapper-import-ok')"

RUN set -eux; \
    printf '#!/usr/bin/env bash\nexec python3 /opt/facefusion/facefusion.py "$@"\n' > /usr/local/bin/facefusion; \
    chmod +x /usr/local/bin/facefusion; \
    test -x /usr/local/bin/facefusion

COPY src /worker/src
WORKDIR /worker/src

# Add model/runtime binaries in a custom image extension:
# - facefusion
# - liveportrait
# - musetalk
# - realesrgan-ncnn-vulkan (optional)

CMD ["python3", "-u", "handler.py"]
