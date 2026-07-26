from __future__ import annotations

import base64
import re
from typing import Any

from .base import ProviderAdapter
from ..core.errors import NoOutputError, ProviderError
from ..core.models import GenerationRequest, ImageResult


_BOOLEAN_PARAMS = {"prompt_extend", "watermark"}
_INTEGER_PARAMS = {"seed"}


def _data_url(mime_type: str, data: bytes) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(data).decode()}"


def _normalize_size(value: str) -> str:
    return re.sub(r"[xX×]", "*", value.strip())


def _parameter_value(key: str, value: str) -> Any:
    if key in _BOOLEAN_PARAMS:
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
        raise ProviderError(f"DashScope 参数 {key} 必须是布尔值")
    if key in _INTEGER_PARAMS:
        try:
            return int(value)
        except ValueError as exc:
            raise ProviderError(f"DashScope 参数 {key} 必须是整数") from exc
    return value


class DashScopeMultimodalAdapter(ProviderAdapter):
    async def generate(self, session, request: GenerationRequest, api_key: str):
        async with session.post(
            self.config.base_url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=self.build_body(request),
        ) as response:
            payload = await self.response_json(response)
        return self.parse_response(payload)

    def build_body(self, request: GenerationRequest) -> dict[str, Any]:
        content = [
            {"image": _data_url(image.mime_type, image.data)}
            for image in request.references
        ]
        content.append({"text": request.prompt})

        parameters: dict[str, Any] = {
            "prompt_extend": True,
            "n": request.count,
        }
        size = request.size or self.config.default_size
        if size:
            parameters["size"] = _normalize_size(size)
        for key, value in request.extra_params.items():
            if key not in {"n", "size"}:
                parameters[key] = _parameter_value(key, value)

        return {
            "model": self.config.model,
            "input": {
                "messages": [{"role": "user", "content": content}],
            },
            "parameters": parameters,
        }

    @staticmethod
    def parse_response(payload: Any) -> list[ImageResult]:
        results: list[ImageResult] = []
        if isinstance(payload, dict):
            output = payload.get("output", {})
            choices = output.get("choices", []) if isinstance(output, dict) else []
            for choice in choices if isinstance(choices, list) else []:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message", {})
                content = message.get("content", []) if isinstance(message, dict) else []
                for item in content if isinstance(content, list) else []:
                    image = item.get("image") if isinstance(item, dict) else None
                    if isinstance(image, str) and image:
                        results.append(ImageResult(url=image))
            if not results and payload.get("code"):
                from ..core.security import redact_debug

                code = redact_debug(str(payload.get("code", "")))
                message = redact_debug(str(payload.get("message", "")))
                detail = f" code={code}" if code else ""
                if message:
                    detail += f" message={message}"
                raise ProviderError(f"DashScope{detail}")
        if not results:
            raise NoOutputError("DashScope 响应中没有可识别的图片")
        return results
