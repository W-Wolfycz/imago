from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .models import ProviderConfig, QuotaConfig, RuntimeConfig

API_TYPES = {"openai_image", "openai_chat", "gemini_official", "dashscope_multimodal", "custom_endpoint"}
STYLES = {"none", "default", "realistic", "cinematic", "anime", "3d"}
STYLE_OPTIONS = {
    "None(无)": "none",
    "default(通用)": "default",
    "realistic(写实)": "realistic",
    "cinematic(电影感)": "cinematic",
    "anime(动漫)": "anime",
    "3d(3D渲染)": "3d",
}


def _line_items(value: Any) -> tuple[str, ...]:
    values = value if isinstance(value, list) else str(value).splitlines()
    return tuple(str(item).strip() for item in values if str(item).strip())


def _keys(value: Any) -> tuple[str, ...]:
    return _line_items(value)


def _ids(value: Any) -> frozenset[str]:
    return frozenset(_line_items(value))


def _style(value: Any) -> str:
    text = str(value).strip()
    if text in STYLES:
        return text
    return STYLE_OPTIONS.get(text, "default")


def _int_or_default(value: Any, default: int) -> int:
    """配置整数兜底：None/非数字回退默认；0 等合法值原样保留（调用方再 clamp）。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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
    style = _style(optimizer.get("optimizer_style", "default(通用)"))
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
        reference_caption=bool(optimizer.get("reference_caption", False)),
        optimizer_prompt=str(optimizer.get("optimizer_prompt", "")).strip(),
        optimizer_style=style,
        fallback_style_injection=bool(optimizer.get("fallback_style_injection", False)),
        generation_timeout=max(30, _int_or_default(tasks.get("generation_timeout"), 300)),
        max_concurrent_tasks=max(1, _int_or_default(tasks.get("max_concurrent_tasks"), 2)),
        llm_retry=max(1, min(5, int(tasks.get("llm_retry", 1) or 1))),
        llm_caption=bool(tasks.get("llm_caption", False)),
        llm_caption_cm_context=bool(tasks.get("llm_caption_cm_context", False)),
        llm_caption_pregen=bool(tasks.get("llm_caption_pregen", False)),
        max_upload_bytes=max(1, int(storage.get("max_upload_mb", 20) or 20)) * 1024 * 1024,
        temp_cache_bytes=max(16, int(storage.get("temp_cache_mb", 512) or 512)) * 1024 * 1024,
        block_private_networks=bool(storage.get("block_private_networks", True)),
        log_with_bot_id=bool(raw.get("log_with_bot_id", False)),
    )


def persona_provider_settings(umo_config: Any, default_config: Any) -> dict:
    """人设解析用的 provider_settings，与主链 LLM 请求的实际数据源对齐。

    AstrBot 4.27.x 主链 `_decorate_llm_request`：

        cfg = config.provider_settings or context.get_config(umo=...).get("provider_settings", {})

    `config.provider_settings` 是 pipeline 按 UMO 路由（umop_config_routing）命中的
    配置文件快照，回退分支 `get_config(umo)` 同样是会话命中配置（acm.get_conf 未命中
    时才内部回退默认配置）——两个分支都不读全局默认配置。多配置文件场景下默认配置与
    会话命中配置的 default_personality 不同，插件侧必须用同一数据源，否则 Persona 任务
    会解析成默认配置的人设（如本插件历史上「全局优先」导致的错位）。

    等价实现：优先 umo_config（会话命中配置），仅当其为 None（调用方未取到）时才
    回退 default_config；会话命中配置存在但 provider_settings 为空时返回空 dict，
    与主链行为一致（不吞回全局默认）。
    """
    source = umo_config if umo_config is not None else default_config
    return (source or {}).get("provider_settings", {}) or {}
