from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


CONTRACT_VERSION = 1


class ContractError(ValueError):
    pass


def _require_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{key} must be a non-empty string")
    return value


def _optional_string(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ContractError(f"{key} must be a string when provided")
    stripped = value.strip()
    return stripped or None


def _require_int(data: dict[str, Any], key: str, minimum: int = 0) -> int:
    value = data.get(key)
    if not isinstance(value, int) or value < minimum:
        raise ContractError(f"{key} must be an integer >= {minimum}")
    return value


def _require_list(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ContractError(f"{key} must be a list")
    return value


@dataclass(slots=True)
class IdentityImage:
    path: str
    name: str
    sha256: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IdentityImage":
        return cls(
            path=_require_string(data, "path"),
            name=_require_string(data, "name"),
            sha256=_require_string(data, "sha256"),
        )


@dataclass(slots=True)
class IdentityPack:
    version: int
    primary_image: str | None
    images: list[IdentityImage]
    identity_video: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "primary_image": self.primary_image,
            "images": [asdict(image) for image in self.images],
            "identity_video": self.identity_video,
        }

    def validate(self) -> None:
        if self.version != CONTRACT_VERSION:
            raise ContractError(f"unsupported identity pack version: {self.version}")
        if not self.images:
            raise ContractError("identity pack requires at least one image")
        image_paths = {image.path for image in self.images}
        if self.primary_image and self.primary_image not in image_paths:
            raise ContractError("primary_image must be one of the images in the identity pack")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IdentityPack":
        version = _require_int(data, "version", minimum=1)
        primary_image = _optional_string(data, "primary_image")
        images = [IdentityImage.from_dict(item) for item in _require_list(data, "images")]
        identity_video = _optional_string(data, "identity_video")
        pack = cls(
            version=version,
            primary_image=primary_image,
            images=images,
            identity_video=identity_video,
        )
        pack.validate()
        return pack


@dataclass(slots=True)
class RenderProfile:
    quality: str
    aspect_ratio: str
    fps: int
    resolution: list[int]
    frame_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.quality not in {"fast", "balanced", "max"}:
            raise ContractError(f"unsupported quality: {self.quality}")
        if self.aspect_ratio not in {"9:16", "1:1", "4:5"}:
            raise ContractError(f"unsupported aspect ratio: {self.aspect_ratio}")
        if self.fps <= 0:
            raise ContractError("render_profile.fps must be > 0")
        if len(self.resolution) != 2 or any(not isinstance(value, int) or value <= 0 for value in self.resolution):
            raise ContractError("render_profile.resolution must contain two positive integers")
        if self.frame_count <= 0:
            raise ContractError("render_profile.frame_count must be > 0")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RenderProfile":
        resolution = _require_list(data, "resolution")
        profile = cls(
            quality=_require_string(data, "quality"),
            aspect_ratio=_require_string(data, "aspect_ratio"),
            fps=_require_int(data, "fps", minimum=1),
            resolution=resolution,
            frame_count=_require_int(data, "frame_count", minimum=1),
        )
        profile.validate()
        return profile


@dataclass(slots=True)
class ControlBundle:
    version: int
    driving_video: str
    frame_dir: str
    sampled_frames: int
    sample_fps: float
    duration_seconds: float | None
    source_fps: float | None
    motion_type: str | None = None
    motion_summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.version != CONTRACT_VERSION:
            raise ContractError(f"unsupported control bundle version: {self.version}")
        if not self.driving_video.strip():
            raise ContractError("control bundle driving_video must be non-empty")
        if not self.frame_dir.strip():
            raise ContractError("control bundle frame_dir must be non-empty")
        if self.sampled_frames <= 0:
            raise ContractError("control bundle sampled_frames must be > 0")
        if self.sample_fps <= 0:
            raise ContractError("control bundle sample_fps must be > 0")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ControlBundle":
        duration_seconds = data.get("duration_seconds")
        source_fps = data.get("source_fps")
        if duration_seconds is not None and not isinstance(duration_seconds, (int, float)):
            raise ContractError("control bundle duration_seconds must be numeric when provided")
        if source_fps is not None and not isinstance(source_fps, (int, float)):
            raise ContractError("control bundle source_fps must be numeric when provided")
        bundle = cls(
            version=_require_int(data, "version", minimum=1),
            driving_video=_require_string(data, "driving_video"),
            frame_dir=_require_string(data, "frame_dir"),
            sampled_frames=_require_int(data, "sampled_frames", minimum=1),
            sample_fps=float(data.get("sample_fps", 0.0)),
            duration_seconds=float(duration_seconds) if duration_seconds is not None else None,
            source_fps=float(source_fps) if source_fps is not None else None,
            motion_type=_optional_string(data, "motion_type"),
            motion_summary=_optional_string(data, "motion_summary"),
        )
        bundle.validate()
        return bundle


@dataclass(slots=True)
class ShotPlan:
    version: int
    task_type: str
    identity_pack_path: str
    control_bundle_path: str | None
    prompt: str
    negative_prompt: str | None
    motion_preset: str | None
    style_preset: str | None
    duration_seconds: int
    seed: int | None
    motion_reference_video: str | None
    motion_reference_profile: dict[str, Any] | None
    driving_audio: str | None
    render_profile: RenderProfile

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "task_type": self.task_type,
            "identity_pack_path": self.identity_pack_path,
            "control_bundle_path": self.control_bundle_path,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "motion_preset": self.motion_preset,
            "style_preset": self.style_preset,
            "duration_seconds": self.duration_seconds,
            "seed": self.seed,
            "motion_reference_video": self.motion_reference_video,
            "motion_reference_profile": self.motion_reference_profile,
            "driving_audio": self.driving_audio,
            "render_profile": self.render_profile.to_dict(),
        }

    def validate(self) -> None:
        if self.version != CONTRACT_VERSION:
            raise ContractError(f"unsupported shot plan version: {self.version}")
        if self.task_type not in {"portrait_reenactment", "image_to_video_generation"}:
            raise ContractError(f"unsupported task_type: {self.task_type}")
        if not self.prompt.strip():
            if self.task_type != "portrait_reenactment":
                raise ContractError("shot plan prompt must be non-empty")
        if self.duration_seconds <= 0:
            raise ContractError("duration_seconds must be > 0")
        if self.motion_reference_profile is not None and not isinstance(self.motion_reference_profile, dict):
            raise ContractError("motion_reference_profile must be a dict when provided")
        self.render_profile.validate()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShotPlan":
        seed = data.get("seed")
        if seed is not None and not isinstance(seed, int):
            raise ContractError("seed must be an integer when provided")
        prompt = data.get("prompt")
        if not isinstance(prompt, str):
            raise ContractError("prompt must be a string")
        plan = cls(
            version=_require_int(data, "version", minimum=1),
            task_type=_require_string(data, "task_type"),
            identity_pack_path=_require_string(data, "identity_pack_path"),
            control_bundle_path=_optional_string(data, "control_bundle_path"),
            prompt=prompt,
            negative_prompt=_optional_string(data, "negative_prompt"),
            motion_preset=_optional_string(data, "motion_preset"),
            style_preset=_optional_string(data, "style_preset"),
            duration_seconds=_require_int(data, "duration_seconds", minimum=1),
            seed=seed,
            motion_reference_video=_optional_string(data, "motion_reference_video"),
            motion_reference_profile=data.get("motion_reference_profile"),
            driving_audio=_optional_string(data, "driving_audio"),
            render_profile=RenderProfile.from_dict(data.get("render_profile", {})),
        )
        plan.validate()
        return plan


@dataclass(slots=True)
class AdapterReport:
    version: int
    stage: str
    engine: str
    model: str
    metrics: dict[str, float | int | str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.version != CONTRACT_VERSION:
            raise ContractError(f"unsupported adapter report version: {self.version}")
        if self.stage not in {"generating", "identity_refine"}:
            raise ContractError(f"unsupported adapter stage: {self.stage}")
        if not self.engine.strip():
            raise ContractError("adapter report engine must be non-empty")
        if not self.model.strip():
            raise ContractError("adapter report model must be non-empty")
        if not isinstance(self.metrics, dict):
            raise ContractError("adapter report metrics must be a dict")
        if not isinstance(self.warnings, list) or any(not isinstance(item, str) for item in self.warnings):
            raise ContractError("adapter report warnings must be a list of strings")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdapterReport":
        report = cls(
            version=_require_int(data, "version", minimum=1),
            stage=_require_string(data, "stage"),
            engine=_require_string(data, "engine"),
            model=_require_string(data, "model"),
            metrics=data.get("metrics", {}),
            warnings=data.get("warnings", []),
        )
        report.validate()
        return report


def load_identity_pack(path: Path) -> IdentityPack:
    return IdentityPack.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_identity_pack(path: Path, pack: IdentityPack) -> None:
    pack.validate()
    path.write_text(json.dumps(pack.to_dict(), indent=2), encoding="utf-8")


def load_shot_plan(path: Path) -> ShotPlan:
    return ShotPlan.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_shot_plan(path: Path, plan: ShotPlan) -> None:
    plan.validate()
    path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")


def load_control_bundle(path: Path) -> ControlBundle:
    return ControlBundle.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_control_bundle(path: Path, bundle: ControlBundle) -> None:
    bundle.validate()
    path.write_text(json.dumps(bundle.to_dict(), indent=2), encoding="utf-8")


def load_adapter_report(path: Path) -> AdapterReport:
    return AdapterReport.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_adapter_report(path: Path, report: AdapterReport) -> None:
    report.validate()
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def ensure_video_output(path: Path) -> None:
    if not path.exists():
        raise ContractError(f"expected output video was not created: {path}")
    if path.stat().st_size <= 0:
        raise ContractError(f"output video is empty: {path}")
