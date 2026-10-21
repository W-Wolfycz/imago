from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod
from typing import Any

from ..core.errors import NoOutputError, UnsupportedResponse
from ..core.models import GenerationRequest, ImageResult, ProviderConfig


def _json_scalar(text: str) -> Any:
    """把 ``true``/``false``/``null``/数字/JSON 文本还原成对应类型，其它保持字符串。

    节点「自定义参数」写的都是文本（``quality=1k``、``nsfw=true``、``size=3:4``），
    发 JSON 时要让布尔和数字回到原生类型，否则上游会按字符串校验失败；``1k``、``3:4``
    这类不是合法 JSON，原样当字符串。
    """
    stripped = (text or "").strip()
    if not stripped:
        return ""
    try:
        return json.loads(stripped)
    except (TypeError, ValueError):
        return stripped


class ProviderAdapter(ABC):
    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    async def generate(self, session: Any, request: GenerationRequest, api_key: str) -> list[ImageResult]: ...

    def custom_params(self) -> dict[str, Any]:
        """节点自定义参数（含 size）：每轮请求都带，值按 JSON 标量还原类型。"""
        return {key: _json_scalar(raw) for key, raw in self.config.custom_params}

    def request_size(self, request: GenerationRequest) -> str:
        """本轮使用哪个尺寸：工具/主 LLM 显式传入优先，其次节点自定义参数里的 size。

        都为空时返回空串，调用方据此不发送 size 字段（交上游自选，不兜底成默认值）。
        """
        size = str(request.size or "").strip()
        if size:
            return size
        return self.config.param("size").strip()

    @staticmethod
    def form_value(value: Any) -> str:
        """multipart 表单值：布尔转 ``true``/``false``，其它字符串化（表单只认文本）。"""
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def parse_common(payload: Any) -> list[ImageResult]:
        results: list[ImageResult] = []
        candidates = payload.get("data", []) if isinstance(payload, dict) else []
        if isinstance(candidates, dict):
            candidates = [candidates]
        for item in candidates:
            if not isinstance(item, dict):
                continue
            url = item.get("url") or item.get("image_url")
            encoded = item.get("b64_json") or item.get("base64")
            if isinstance(url, str) and url:
                results.append(ImageResult(url=url))
            elif isinstance(encoded, str) and encoded:
                try:
                    results.append(ImageResult(data=base64.b64decode(encoded, validate=True)))
                except ValueError:
                    continue
        if not results:
            raise NoOutputError("响应中没有可识别的图片")
        return results

    @staticmethod
    def _error_detail(payload: Any) -> str:
        """从错误响应体提取可诊断摘要。

        兼容三种常见形态：顶层字段（百炼 code/message）、OpenAI/Gemini 的嵌套
        ``error`` 对象（message/type/code）、``error`` 为纯字符串的中转站。
        全部经 redact_debug 脱敏并截断，避免把完整响应体打进日志。
        """
        from ..core.security import redact_debug

        if not isinstance(payload, dict):
            return ""
        error = payload.get("error")
        if isinstance(error, dict):
            merged = dict(payload)
            merged.update(error)
            payload = merged
        elif isinstance(error, str) and error.strip():
            return f"message={redact_debug(error)}"
        parts = []
        for key, label in (
            ("code", "code"),
            ("message", "message"),
            ("msg", "message"),
            ("type", "type"),
            ("err", "message"),
        ):
            value = payload.get(key)
            if value is not None and str(value).strip():
                parts.append(f"{label}={redact_debug(str(value))[:160]}")
        return " ".join(parts)

    @staticmethod
    async def response_json(response: Any) -> Any:
        if response.status >= 400:
            from ..core.errors import ProviderError
            from ..core.security import redact_debug
            detail = ""
            try:
                payload = await response.json(content_type=None)
                detail = ProviderAdapter._error_detail(payload)
            except Exception:
                payload = None
            if not detail:
                # 非 JSON 或未知结构：截断原始文本兜底，保证日志仍可诊断。
                try:
                    raw = await response.text()
                except Exception:
                    raw = ""
                if raw and str(raw).strip():
                    detail = "body=" + redact_debug(str(raw).strip())[:160]
            raise ProviderError(f"HTTP {response.status}{' ' + detail if detail else ''}", status=response.status)
        try:
            return await response.json(content_type=None)
        except Exception as exc:
            raise UnsupportedResponse("上游返回的不是有效 JSON") from exc
