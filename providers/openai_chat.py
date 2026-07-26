from __future__ import annotations

import base64
import re
from urllib.parse import urljoin

from .base import ProviderAdapter
from ..core.errors import NoOutputError, UnsupportedResponse
from ..core.models import GenerationRequest, ImageResult


class OpenAIChatAdapter(ProviderAdapter):
    async def generate(self, session, request: GenerationRequest, api_key: str):
        content = request.prompt
        if request.references:
            content = [{"type": "text", "text": request.prompt}]
            content.extend({"type": "image_url", "image_url": {"url": f"data:{image.mime_type};base64,{base64.b64encode(image.data).decode()}"}} for image in request.references)
        body = {"model": self.config.model, "messages": [{"role": "user", "content": content}], "n": request.count, **request.extra_params}
        url = urljoin(self.config.base_url.rstrip("/") + "/", "chat/completions")
        async with session.post(url, headers={"Authorization": f"Bearer {api_key}"}, json=body) as response:
            payload = await self.response_json(response)
        if not isinstance(payload, dict):
            raise UnsupportedResponse("Chat 上游响应格式无效")
        results: list[ImageResult] = []
        choices = payload.get("choices", [])
        if not isinstance(choices, list):
            raise UnsupportedResponse("Chat 上游 choices 格式无效")
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message", {})
            if not isinstance(message, dict):
                continue
            images = message.get("images", []) or []
            if not isinstance(images, list):
                images = []
            for item in images:
                if not isinstance(item, (dict, str)):
                    continue
                value = item.get("image_url", item) if isinstance(item, dict) else item
                value = value.get("url", "") if isinstance(value, dict) else value
                if isinstance(value, str): results.append(ImageResult(url=value))
            text = message.get("content", "")
            if isinstance(text, list):
                for part in text:
                    if isinstance(part, dict) and part.get("type") in {"image", "image_url"}:
                        value = part.get("image_url") or part.get("url") or part.get("data")
                        if isinstance(value, dict): value = value.get("url")
                        if value: results.append(ImageResult(url=value))
            elif isinstance(text, str):
                for value in re.findall(r"(?:https?://[^\s)]+|data:image/[^;]+;base64,[A-Za-z0-9+/=]+)", text):
                    results.append(ImageResult(url=value))
        if not results:
            raise NoOutputError("Chat 响应中没有图片")
        return results
