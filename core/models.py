from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class TaskState(str, Enum):
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL_SUCCESS = "partial_success"
    DELIVERY_FAILED = "delivery_failed"
    NO_OUTPUT = "no_output"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class TaskStage(str, Enum):
    QUEUED = "queued"
    PREPARING_REFERENCES = "preparing_references"
    BUILDING_PERSONA = "building_persona"
    OPTIMIZING_PROMPT = "optimizing_prompt"
    REQUESTING_PROVIDER = "requesting_provider"
    PROCESSING_RESULT = "processing_result"
    DECORATING = "decorating"
    SENDING = "sending"
    FINISHED = "finished"


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    id: str
    api_type: str
    base_url: str
    api_keys: tuple[str, ...]
    model: str = ""
    available_models: tuple[str, ...] = ()
    reference_image_limit: int = 0
    default_size: str = "1024x1024"
    timeout: int = 180


@dataclass(frozen=True, slots=True)
class QuotaConfig:
    enabled: bool = False
    blacklist_ids: frozenset[str] = frozenset()
    unlimited_whitelist_ids: frozenset[str] = frozenset()
    daily_refresh_enabled: bool = True
    daily_quota_target: int = 0
    checkin_enabled: bool = False
    checkin_quota_min: int = 1
    checkin_quota_max: int = 3


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    providers: tuple[ProviderConfig, ...]
    quota: QuotaConfig = field(default_factory=QuotaConfig)
    optimizer_enabled: bool = True
    optimizer_provider_id: str = ""
    vision_provider_id: str = ""
    optimizer_prompt: str = ""
    optimizer_style: str = "default"
    fallback_style_injection: bool = False
    generation_timeout: int = 300
    max_concurrent_tasks: int = 2
    llm_caption: bool = False
    llm_caption_cm_context: bool = False
    llm_caption_pregen: bool = False
    max_upload_bytes: int = 20 * 1024 * 1024
    temp_cache_bytes: int = 512 * 1024 * 1024
    block_private_networks: bool = True
    log_with_bot_id: bool = False


@dataclass(slots=True)
class ImageInput:
    data: bytes
    mime_type: str
    filename: str = "reference"


@dataclass(slots=True)
class ImageResult:
    url: str = ""
    data: bytes | None = None
    mime_type: str = "image/png"
    local_path: Path | None = None


@dataclass(slots=True)
class GenerationRequest:
    prompt: str
    count: int = 1
    size: str = ""
    aspect_ratio: str = ""
    references: list[ImageInput] = field(default_factory=list)
    extra_params: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class DrawTask:
    id: str
    umo: str
    request: GenerationRequest
    persona_id: str = ""
    persona_prompt: str = ""
    owner_user_id: str = ""
    bot_instance_id: str = ""
    kind: str = "draw"
    state: TaskState = TaskState.CREATED
    stage: TaskStage = TaskStage.QUEUED
    created_at: float = 0.0
    updated_at: float = 0.0
    errors: list[str] = field(default_factory=list)
    runtime: dict[str, Any] = field(default_factory=dict)
