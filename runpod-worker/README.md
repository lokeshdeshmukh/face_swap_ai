# Runpod Worker

This worker expects the backend payload contract documented in the root README.

## Build

```bash
docker build -t truefaceswap-runpod-worker .
```

## Deploy

Push the image to a registry and configure it as the Runpod Serverless endpoint image.

## Important

- This template assumes model CLIs are present in the image.
- For production-sized outputs, replace `output_base64` with object-storage upload and return an `output_url`.
