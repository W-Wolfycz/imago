from __future__ import annotations

import base64
import binascii
import math
import re
from urllib.parse import quote, urlsplit

from .base import ProviderAdapter
from ..core.errors import NoOutputError, UnsupportedResponse
from ..core.models import GenerationRequest, ImageResult

_PIXELS = re.compile(r"^(\d+)\s*[x*]\s*(\d+)$", re.IGNORECASE)
_RATIO = re.compile(r"^\d+(?:\.\d+)?:\d+(?:\.\d+)?$")


def _aspect_ratio(size: str) -> str:
    """把 size 归一成 Gemini 的 ``aspectRatio``：比例串原样，像素尺寸约分。"""
    text = str(size or "").strip()
    if _RATIO.fullmatch(text):
        return text
    match = _PIXELS.fullmatch(text)
    if not match:
        return ""
    width, height = int(match.group(1)), int(match.group(2))
    if not width or not height:
        return ""
    divisor = math.gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def _file_uri_with_key(uri: str, api_key: str, api_host: str = "") -> str:
    """Files API 下载 URI 需要携带 API key；只在**同一服务商**的地址上追加。

    中转常把结果图放到自己的对象存储（另一个域名）。把 key 拼到第三方 URL 上等于把
    密钥交给对方日志，所以只对与 base_url 同域的 URI 补 key，其余原样返回。
    """
    uri = str(uri or "")
    if not uri or not api_key or "key=" in uri:
        return uri
    try:
        same_host = bool(api_host) and urlsplit(uri).netloc.lower() == api_host.lower()
    except ValueError:
        same_host = False
    if not same_host:
        return uri
    return f"{uri}{'&' if '?' in uri else '?'}key={quote(api_key, safe='')}"


class GeminiAdapter(ProviderAdapter):
    async def generate(self, session, request: GenerationRequest, api_key: str):
        parts = [{"text": request.prompt}]
        parts.extend({"inlineData": {"mimeType": image.mime_type, "data": base64.b64encode(image.data).decode()} for image in request.references})
        # 尺寸：Gemini 原生用 generationConfig.imageConfig.aspectRatio；像素尺寸先约分。
        image_config = {key: value for key, value in self.custom_params().items() if key != "size"}
        if "imageSize" in image_config:
            # 官方 schema 里 imageSize 是字符串（512/1K/2K/4K），自定义参数按 JSON 解析会变数字
            image_config["imageSize"] = str(image_config["imageSize"])
        ratio = _aspect_ratio(self.request_size(request))
        if ratio and "aspectRatio" not in image_config:
            image_config["aspectRatio"] = ratio
        generation: dict = {"responseModalities": ["TEXT", "IMAGE"]}
        if image_config:
            generation["imageConfig"] = image_config
        body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": generation}
        base = self.config.base_url.rstrip("/")
        url = f"{base}/models/{quote(self.config.model, safe='')}:generateContent?key={quote(api_key, safe='')}"
        async with session.post(url, json=body) as response:
            payload = await self.response_json(response)
        if not isinstance(payload, dict):
            raise UnsupportedResponse("Gemini 上游响应格式无效")
        api_host = urlsplit(self.config.base_url).netloc
        results = []
        candidates = payload.get("candidates", [])
        if not isinstance(candidates, list):
            raise UnsupportedResponse("Gemini candidates 格式无效")
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content", {})
            parts = content.get("parts", []) if isinstance(content, dict) else []
            for part in parts if isinstance(parts, list) else []:
                if not isinstance(part, dict):
                    continue
                inline = part.get("inlineData") or part.get("inline_data")
                if isinstance(inline, dict) and inline.get("data"):
                    try:
                        data = base64.b64decode(inline["data"], validate=True)
                    except (ValueError, binascii.Error, TypeError):
                        continue
                    results.append(ImageResult(data=data, mime_type=inline.get("mimeType", inline.get("mime_type", "image/png"))))
                else:
                    file_data = part.get("fileData", {})
                    if isinstance(file_data, dict) and file_data.get("fileUri"):
                        # 同域的 Files API 需要 key；异域（对象存储/CDN）不拼，避免泄露密钥。
                        results.append(ImageResult(url=_file_uri_with_key(str(file_data["fileUri"]), api_key, api_host)))
        if not results: raise NoOutputError("Gemini 响应中没有图片")
        return results
