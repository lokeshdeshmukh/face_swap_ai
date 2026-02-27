FROM runpod/base:0.4.1-cuda12.1.0

WORKDIR /worker

RUN apt-get update && \
    apt-get install -y --no-install-recommends git ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /worker/requirements.txt
RUN set -eux; \
    ln -sf "$(command -v python3)" /usr/local/bin/python; \
    python3 -m ensurepip --upgrade || true; \
    python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel; \
    python3 -m pip install --no-cache-dir -r /worker/requirements.txt; \
    python3 -c "import requests, runpod; print('deps-ok')"

RUN git clone --depth 1 https://github.com/facefusion/facefusion.git /opt/facefusion && \
    python3 -m pip install --no-cache-dir -r /opt/facefusion/requirements.txt && \
    printf '#!/usr/bin/env bash\nexec python3 /opt/facefusion/facefusion.py "$@"\n' > /usr/local/bin/facefusion && \
    chmod +x /usr/local/bin/facefusion

COPY src /worker/src
WORKDIR /worker/src

# Add model/runtime binaries in a custom image extension:
# - facefusion
# - liveportrait
# - musetalk
# - realesrgan-ncnn-vulkan (optional)

CMD ["python3", "-u", "handler.py"]
