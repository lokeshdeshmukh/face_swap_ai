from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import sys
import types
import urllib.request
from urllib.parse import unquote, urlparse


def _install_torchvision_functional_tensor_alias() -> None:
    module_name = "torchvision.transforms.functional_tensor"
    if module_name in sys.modules:
        return

    try:
        import torchvision.transforms._functional_tensor as functional_tensor  # type: ignore[attr-defined]
    except Exception:
        return

    shim = types.ModuleType(module_name)
    for attribute in dir(functional_tensor):
        if attribute.startswith("__"):
            continue
        setattr(shim, attribute, getattr(functional_tensor, attribute))
    sys.modules[module_name] = shim


_install_torchvision_functional_tensor_alias()


def _install_huggingface_hub_cached_download_alias() -> None:
    try:
        import huggingface_hub
    except Exception:
        return

    if hasattr(huggingface_hub, "cached_download") or not hasattr(huggingface_hub, "hf_hub_download"):
        return

    def _cached_download(
        url_or_filename: str,
        *args,
        cache_dir: str | os.PathLike[str] | None = None,
        force_filename: str | None = None,
        token: str | bool | None = None,
        use_auth_token: str | bool | None = None,
        local_files_only: bool = False,
        force_download: bool = False,
        **kwargs,
    ) -> str:
        candidate = Path(url_or_filename)
        if candidate.exists():
            return str(candidate)

        resolved_cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "huggingface" / "hub"
        resolved_cache_dir.mkdir(parents=True, exist_ok=True)

        parsed = urlparse(url_or_filename)
        path_segments = [segment for segment in parsed.path.split("/") if segment]
        auth_token = token if token is not None else use_auth_token
        if (
            parsed.scheme in {"http", "https"}
            and parsed.netloc.endswith("huggingface.co")
            and "resolve" in path_segments
        ):
            resolve_index = path_segments.index("resolve")
            repo_type = None
            repo_start = 0
            if path_segments[0] in {"models", "datasets", "spaces"}:
                repo_start = 1
                if path_segments[0] != "models":
                    repo_type = path_segments[0][:-1]

            repo_id = "/".join(path_segments[repo_start:resolve_index])
            revision = path_segments[resolve_index + 1]
            filename = unquote("/".join(path_segments[resolve_index + 2 :]))
            return huggingface_hub.hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                cache_dir=str(resolved_cache_dir),
                token=auth_token,
                local_files_only=local_files_only,
                repo_type=repo_type,
            )

        destination_name = force_filename or Path(parsed.path).name or hashlib.sha256(url_or_filename.encode("utf-8")).hexdigest()
        destination = resolved_cache_dir / destination_name
        if destination.exists() and not force_download:
            return str(destination)

        headers: dict[str, str] = {}
        if isinstance(auth_token, str) and auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        request = urllib.request.Request(url_or_filename, headers=headers)
        with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        return str(destination)

    huggingface_hub.cached_download = _cached_download


_install_huggingface_hub_cached_download_alias()
