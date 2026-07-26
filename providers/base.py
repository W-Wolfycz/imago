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
    async def response_json(response: Any) -> Any:
        if response.status >= 400:
            from ..core.errors import ProviderError
            from ..core.security import redact_debug
            detail = ""
            try:
                payload = await response.json(content_type=None)
                if isinstance(payload, dict):
                    code = str(payload.get("code", "")).strip()
                    message = str(payload.get("message", "")).strip()
                    parts = []
                    if code:
                        parts.append(f"code={redact_debug(code)}")
                    if message:
                        parts.append(f"message={redact_debug(message)}")
                    detail = " " + " ".join(parts) if parts else ""
            except Exception:
                pass
            raise ProviderError(f"HTTP {response.status}{detail}", status=response.status)
        try:
            return await response.json(content_type=None)
        except Exception as exc:
            raise UnsupportedResponse("上游返回的不是有效 JSON") from exc
