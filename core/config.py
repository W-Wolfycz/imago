from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .models import ProviderConfig, QuotaConfig, RuntimeConfig

API_TYPES = {"openai_image", "openai_chat", "gemini_official", "dashscope_multimodal", "custom_endpoint"}
STYLES = {"default", "realistic", "cinematic", "anime", "3d"}


def _line_items(value: Any) -> tuple[str, ...]:
    values = value if isinstance(value, list) else str(value).splitlines()
    return tuple(str(item).strip() for item in values if str(item).strip())


def _keys(value: Any) -> tuple[str, ...]:
    return _line_items(value)


def _ids(value: Any) -> frozenset[str]:
    return frozenset(_line_items(value))


def load_config(raw: Mapping[str, Any]) -> RuntimeConfig:
    providers: list[ProviderConfig] = []
    seen: set[str] = set()
    for item in raw.get("providers", []) or []:
        if not isinstance(item, Mapping):
            continue
        provider_id = str(item.get("id", "")).strip()
        api_type = str(item.get("api_type", "")).strip()
        base_url = str(item.get("base_url", "")).strip()
        api_keys = _keys(item.get("api_keys", ""))
        if not provider_id or provider_id in seen or api_type not in API_TYPES or not base_url or not api_keys:
            continue
        seen.add(provider_id)
        providers.append(ProviderConfig(
            id=provider_id,
            api_type=api_type,
            base_url=base_url,
            api_keys=api_keys,
            model=str(item.get("model", "")).strip(),
            available_models=tuple(str(v) for v in (item.get("available_models", []) or [])),
            reference_image_limit=max(0, int(item.get("reference_image_limit", 0) or 0)),
            default_size=str(item.get("default_size", "1024x1024")).strip() or "1024x1024",
            timeout=max(10, int(item.get("timeout", 180) or 180)),
        ))
    optimizer = raw.get("optimizer_config", {}) or {}
    quota_raw = raw.get("quota_config", {}) or {}
    tasks = raw.get("task_config", {}) or {}
    storage = raw.get("storage_config", {}) or {}
    logs = raw.get("log_config", {}) or {}
    style = str(optimizer.get("optimizer_style", "default"))
    checkin_min = max(0, int(quota_raw.get("daily_checkin_quota_min", 1) or 0))
    checkin_max = max(checkin_min, int(quota_raw.get("daily_checkin_quota_max", 3) or 0))
    return RuntimeConfig(
        providers=tuple(providers),
        quota=QuotaConfig(
            enabled=bool(quota_raw.get("enable_quota", False)),
            blacklist_ids=_ids(quota_raw.get("blacklist_ids", [])),
            unlimited_whitelist_ids=_ids(quota_raw.get("unlimited_whitelist_ids", [])),
            daily_refresh_enabled=bool(quota_raw.get("enable_daily_refresh", True)),
            daily_quota_target=max(0, int(quota_raw.get("daily_quota_target", 0) or 0)),
            checkin_enabled=bool(quota_raw.get("enable_checkin", False)),
            checkin_quota_min=checkin_min,
            checkin_quota_max=checkin_max,
        ),
        optimizer_enabled=bool(optimizer.get("enable_optimizer", True)),
        optimizer_provider_id=str(optimizer.get("optimizer_provider_id", "")).strip(),
        vision_provider_id=str(optimizer.get("vision_provider_id", "")).strip(),
        optimizer_prompt=str(optimizer.get("optimizer_prompt", "")).strip(),
        optimizer_style=style if style in STYLES else "default",
        generation_timeout=max(30, int(tasks.get("generation_timeout", 300))),
        max_concurrent_tasks=max(1, int(tasks.get("max_concurrent_tasks", 2))),
        max_upload_bytes=max(1, int(storage.get("max_upload_mb", 20) or 20)) * 1024 * 1024,
        temp_cache_bytes=max(16, int(storage.get("temp_cache_mb", 512) or 512)) * 1024 * 1024,
        block_private_networks=bool(storage.get("block_private_networks", True)),
        debug_to_info=bool(logs.get("debug_to_info", False)),
        log_with_bot_id=bool(logs.get("log_with_bot_id", False)),
    )
