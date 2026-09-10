from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .models import ProviderConfig, QuotaConfig, RuntimeConfig

API_TYPES = {"openai_image", "openai_chat", "gemini_official", "dashscope_multimodal", "custom_endpoint"}
STYLES = {"none", "default", "realistic", "real3d", "cg3d", "illustration", "pixel", "logo", "auto"}
STYLE_OPTIONS = {
    "None(无)": "none",
    "default(通用)": "default",
    "realistic(真人实拍)": "realistic",
    "real3d(照片级三维渲染)": "real3d",
    "cg3d(风格化三维渲染)": "cg3d",
    "illustration(手绘插画)": "illustration",
    "pixel(像素阵列)": "pixel",
    "logo(LOGO 设计)": "logo",
    "auto(自动)": "auto",
}


def _line_items(value: Any) -> tuple[str, ...]:
    values = value if isinstance(value, list) else str(value).splitlines()
    return tuple(str(item).strip() for item in values if str(item).strip())


def _keys(value: Any) -> tuple[str, ...]:
    return _line_items(value)


def _ids(value: Any) -> frozenset[str]:
    return frozenset(_line_items(value))


def _style(value: Any) -> str:
    """解析 optimizer_style：现用内部键或 WebUI 标签，其余一律回退 default(通用)。

    不做旧值兼容：1.1.6 之前的标签（`anime(动漫)`、`3d(3D渲染)` 等）不再识别，
    升级后需要重新选一次；旧配置不会被改写，只是解析结果落到 default(通用)。
    """
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
        optimize_plain_draw=bool(optimizer.get("optimize_plain_draw", False)),
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
        at_trigger_user=bool(tasks.get("at_trigger_user", True)),
        max_upload_bytes=max(1, int(storage.get("max_upload_mb", 20) or 20)) * 1024 * 1024,
        temp_cache_bytes=max(16, int(storage.get("temp_cache_mb", 512) or 512)) * 1024 * 1024,
        block_private_networks=bool(storage.get("block_private_networks", True)),
        log_with_bot_id=bool(raw.get("log_with_bot_id", False)),
    )


def optimizer_in_use(cfg: RuntimeConfig, *, persona: bool, plain_draw_tool: bool) -> bool:
    """本轮任务是否调用副脑（纯逻辑，与风格联动完全独立）。

    - `enable_optimizer` 是副脑总开关，关闭时任何路径都不调用；
    - Persona 出镜任务在总开关开启时始终调用；
    - 普通绘图只有来自 LLM 工具（`generate_image`）且 `optimize_plain_draw` 开启
      时才调用——该开关只决定这件事，不参与风格裁决（风格联动见
      `core.prompting.resolve_style`，唯一联动点是配置为 `auto`）。
    """
    if not cfg.optimizer_enabled:
        return False
    if persona:
        return True
    return bool(plain_draw_tool and cfg.optimize_plain_draw)


def style_arg_consumed(cfg: RuntimeConfig, *, use_optimizer: bool) -> bool:
    """主 LLM 传的基准参数是否会被本轮消费（纯逻辑）。

    会被消费的只有两种情形：

    - 本轮副脑参与：参数交给 `core.prompting.resolve_style` 裁决，配置为 auto 时生效；
    - 「副脑降级时注入风格后缀」开启：副脑没跑（或调用失败降级）时也要按它注入后缀。

    其余情形参数被丢弃——副脑没跑又没有降级注入时，解析出来的基准没有任何消费方，
    留着只会让 `/画` 之类的直连路径多一个看不见的状态。
    """
    return bool(use_optimizer or cfg.fallback_style_injection)


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
