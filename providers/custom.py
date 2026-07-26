from __future__ import annotations

import base64

from .base import ProviderAdapter
from ..core.models import GenerationRequest


class CustomEndpointAdapter(ProviderAdapter):
    async def generate(self, session, request: GenerationRequest, api_key: str):
        body = {
            "prompt": request.prompt,
            "model": self.config.model,
            "count": request.count,
            "size": request.size or self.config.default_size,
            "aspect_ratio": request.aspect_ratio,
            "references": [{"mime_type": image.mime_type, "data": base64.b64encode(image.data).decode()} for image in request.references],
            "parameters": request.extra_params,
        }
        async with session.post(self.config.base_url, headers={"Authorization": f"Bearer {api_key}"}, json=body) as response:
            return self.parse_common(await self.response_json(response))
