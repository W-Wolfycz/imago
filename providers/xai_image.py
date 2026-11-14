from __future__ import annotations

import base64
import math
import re
from typing import Any
from urllib.parse import urljoin

from .openai_image import OpenAIImageAdapter
from ..core.models import GenerationRequest, ImageInput

# Grok 的 ratio 写法：`1:1`、`3:4`、`9:19.5`、`auto`
_RATIO = re.compile(r"^(?:\d+(?:\.\d+)?:\d+(?:\.\d+)?|auto)$", re.IGNORECASE)
_PIXELS = re.compile(r"^(\d+)\s*[x*]\s*(\d+)$", re.IGNORECASE)


def _ratio_of(text: str) -> str:
    """把 size 归一成 aspect_ratio：比例串原样，像素尺寸约分，其余返回空串。"""
    value = str(text or "").strip()
    if _RATIO.fullmatch(value):
        return value
    match = _PIXELS.fullmatch(value)
    if not match:
        return ""
    width, height = int(match.group(1)), int(match.group(2))
    if not width or not height:
        return ""
    divisor = math.gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def _data_url(image: ImageInput) -> str:
    return f"data:{image.mime_type};base64,{base64.b64encode(image.data).decode()}"


class XaiImageAdapter(OpenAIImageAdapter):
    """xAI Grok Imagine 生图/改图（``/v1/images/generations`` 与 ``/v1/images/edits``）。

    按官方口径实现，只保证端点与字段名；``model``、``resolution``、``quality``、``nsfw``、
    ``response_format`` 等一律由节点配置与「自定义参数」透传，插件不写死任何一家的私有参数。

    与 ``openai_image`` 的协议差异：
    - **改图只走 JSON**（官方只收 ``application/json``，明确不支持 OpenAI SDK 的 multipart
      ``images.edit()``）：单图放 ``image: {url}``，多图放 ``images: [{url}, …]``（两者互斥），
      url 用 base64 data URL；多图时提示词里可用 ``<IMAGE_0>``、``<IMAGE_1>`` 按顺序指代。
    - **官方没有 ``size`` 字段**，尺寸只由 ``aspect_ratio``（`1:1`/`3:4`/… /`auto`）决定。
      节点或工具里填的 ``size`` 会被折算成比例（像素尺寸约分，如 ``1024x1024`` → ``1:1``），
      不再作为 ``size`` 发给上游。
    """

    def request_fields(self, request: GenerationRequest) -> dict[str, Any]:
        fields = super().request_fields(request)
        # 官方没有 size：折算成 aspect_ratio 后丢弃，避免给严格校验的中转发非法字段
        size = str(fields.pop("size", "") or "")
        if "aspect_ratio" not in fields:
            ratio = _ratio_of(str(request.aspect_ratio or "").strip() or size)
            if ratio:
                fields["aspect_ratio"] = ratio
        return fields

    async def generate(self, session, request: GenerationRequest, api_key: str):
        if not request.references:
            return await super().generate(session, request, api_key)
        fields = self.request_fields(request)
        fields.pop("image", None)
        fields.pop("images", None)
        sources = [{"url": _data_url(image)} for image in request.references]
        if len(sources) == 1:
            fields["image"] = sources[0]
        else:
            fields["images"] = sources
        url = urljoin(self.config.base_url.rstrip("/") + "/", "images/edits")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        async with session.post(url, headers=headers, json=fields) as response:
            return self.parse_common(await self.response_json(response))
