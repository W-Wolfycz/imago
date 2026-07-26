from __future__ import annotations

import json
from urllib.parse import urljoin

from .base import ProviderAdapter
from ..core.models import GenerationRequest


class OpenAIImageAdapter(ProviderAdapter):
    async def generate(self, session, request: GenerationRequest, api_key: str):
        headers = {"Authorization": f"Bearer {api_key}"}
        model = self.config.model
        size = request.size or self.config.default_size
        if not request.references:
            url = urljoin(self.config.base_url.rstrip("/") + "/", "images/generations")
            payload = {"model": model, "prompt": request.prompt, "n": request.count, "size": size, **request.extra_params}
            async with session.post(url, headers={**headers, "Content-Type": "application/json"}, json=payload) as response:
                return self.parse_common(await self.response_json(response))
        url = urljoin(self.config.base_url.rstrip("/") + "/", "images/edits")
        form = __import__("aiohttp").FormData()
        form.add_field("model", model)
        form.add_field("prompt", request.prompt)
        form.add_field("n", str(request.count))
        form.add_field("size", size)
        for key, value in request.extra_params.items():
            form.add_field(key, value)
        for index, image in enumerate(request.references):
            form.add_field("image[]", image.data, filename=image.filename or f"reference-{index}", content_type=image.mime_type)
        async with session.post(url, headers=headers, data=form) as response:
            return self.parse_common(await self.response_json(response))
