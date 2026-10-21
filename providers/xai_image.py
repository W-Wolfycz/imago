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

    各家 Grok 中转的模型 ID 与私有参数不同，插件不写死：``model`` 取节点配置，
    ``resolution``/``quality``/``nsfw`` 之类全部由节点「自定义参数」透传。

    与 ``openai_image`` 的协议差异：
    - **改图默认走 JSON**（官方 xAI 只收 ``application/json``，明确不支持 OpenAI SDK 的
      multipart ``images.edit()``）：单图放 ``image: {url}``，多图放 ``images: [{url}, …]``，
      url 用 base64 data URL，多图时提示词里可用 ``<IMAGE_0>``、``<IMAGE_1>`` 指代。
    - 只支持 multipart 的中转（例如 pipiwang）把节点自定义参数写成 ``edit_mode=multipart``
      即可切回旧方式，表单字段名是 ``image``（不带 ``[]``）。
    - 官方用 ``aspect_ratio``、部分站点只认 ``size``：两个都发。``size`` 原样发送，
      ``aspect_ratio`` 由它折算（像素尺寸约分，如 ``1024x1024`` → ``1:1``）；显式配置或
      工具传入的 ``aspect_ratio`` 优先。
    """

    image_field = "image"  # multipart 方言使用的表单字段名

    def edit_mode(self) -> str:
        """改图传输方式：``json``（默认，官方口径）或 ``multipart``（中转口径）。"""
        value = self.config.param("edit_mode").strip().lower()
        return value if value in {"json", "multipart"} else "json"

    def request_fields(self, request: GenerationRequest) -> dict[str, Any]:
        fields = super().request_fields(request)
        fields.pop("edit_mode", None)  # 本地开关，不发给上游
        # 官方没有 size 字段：同时发 size（中转按它定尺寸/比例）与 aspect_ratio（官方按它定比例）
        size = fields.get("size", "")
        fields.pop("send_size", None)
        if "aspect_ratio" not in fields:
            ratio = _ratio_of(str(request.aspect_ratio or "").strip() or str(size or ""))
            if ratio:
                fields["aspect_ratio"] = ratio
        return fields

    async def generate(self, session, request: GenerationRequest, api_key: str):
        if not request.references or self.edit_mode() == "multipart":
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
