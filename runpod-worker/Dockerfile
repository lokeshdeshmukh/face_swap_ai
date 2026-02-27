FROM runpod/base:0.4.1-cuda12.1.0

WORKDIR /worker

COPY requirements.txt /worker/requirements.txt
RUN pip install --no-cache-dir -r /worker/requirements.txt

COPY src /worker/src
WORKDIR /worker/src

# Add model/runtime binaries in a custom image extension:
# - facefusion
# - liveportrait
# - musetalk
# - realesrgan-ncnn-vulkan (optional)

CMD ["python", "handler.py"]
