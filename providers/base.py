from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from typing import Any

from ..core.errors import NoOutputError, UnsupportedResponse
from ..core.models import GenerationRequest, ImageResult, ProviderConfig


class ProviderAdapter(ABC):
    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    async def generate(self, session: Any, request: GenerationRequest, api_key: str) -> list[ImageResult]: ...

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
