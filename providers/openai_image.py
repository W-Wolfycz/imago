from __future__ import annotations

from typing import Any
from urllib.parse import urljoin

from .base import ProviderAdapter
from ..core.models import GenerationRequest


class OpenAIImageAdapter(ProviderAdapter):
    """OpenAI 兼容的 images 接口：无参考图走 generations（JSON），有参考图走 edits。

    请求字段按「协议默认值 → 节点自定义参数 → 工具附加参数 → 保留字段」合并：
    ``model``/``prompt``/``n``/``size`` 永远由插件按本轮请求写死，自定义参数只补其余字段。
    """

    # 参考图字段名：OpenAI 用 image[]，xAI / Grok 中转用 image（见 XaiImageAdapter）
    image_field = "image[]"
    # 协议固定携带、可被自定义参数覆盖的字段
    default_fields: dict[str, Any] = {}

    def request_fields(self, request: GenerationRequest) -> dict[str, Any]:
        merged = {**self.default_fields, **self.custom_params(), **request.extra_params}
        # 空值不发送：既能避免上游把空字符串当非法参数，也让 `quality=` / `nsfw=` 这种
        # 写法能关掉协议默认字段（Grok 节点用 image-1 时中转站未必认这两个字段）。
        fields = {key: value for key, value in merged.items() if value is not None and value != ""}
        size = self.request_size(request)
        fields.pop("size", None)
        if size:  # 留空/未指定：不发送该字段，交给上游
            fields["size"] = size
        fields["model"] = self.config.model
        fields["prompt"] = request.prompt
        fields["n"] = request.count
        return fields

    async def generate(self, session, request: GenerationRequest, api_key: str):
        headers = {"Authorization": f"Bearer {api_key}"}
        fields = self.request_fields(request)
        if not request.references:
            url = urljoin(self.config.base_url.rstrip("/") + "/", "images/generations")
            async with session.post(url, headers={**headers, "Content-Type": "application/json"}, json=fields) as response:
                return self.parse_common(await self.response_json(response))
        url = urljoin(self.config.base_url.rstrip("/") + "/", "images/edits")
        form = __import__("aiohttp").FormData()
        for key, value in fields.items():
            form.add_field(key, self.form_value(value))
        for index, image in enumerate(request.references):
            form.add_field(self.image_field, image.data, filename=image.filename or f"reference-{index}", content_type=image.mime_type)
        async with session.post(url, headers=headers, data=form) as response:
            return self.parse_common(await self.response_json(response))
