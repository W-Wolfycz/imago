from __future__ import annotations

import base64
import binascii
from urllib.parse import quote

from .base import ProviderAdapter
from ..core.errors import NoOutputError, UnsupportedResponse
from ..core.models import GenerationRequest, ImageResult


def _file_uri_with_key(uri: str, api_key: str) -> str:
    """Files API 下载 URI 需要携带 API key；缺失时按查询参数规则追加。

    已带 key 参数时原样返回；api_key 为空时不追加（可能是服务账号等
    其他鉴权方式的部署）。
    """
    uri = str(uri or "")
    if not uri or not api_key or "key=" in uri:
        return uri
    return f"{uri}{'&' if '?' in uri else '?'}key={quote(api_key, safe='')}"


class GeminiAdapter(ProviderAdapter):
    async def generate(self, session, request: GenerationRequest, api_key: str):
        parts = [{"text": request.prompt}]
        parts.extend({"inlineData": {"mimeType": image.mime_type, "data": base64.b64encode(image.data).decode()}} for image in request.references)
        body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}}
        base = self.config.base_url.rstrip("/")
        url = f"{base}/models/{quote(self.config.model, safe='')}:generateContent?key={quote(api_key, safe='')}"
        async with session.post(url, json=body) as response:
            payload = await self.response_json(response)
        if not isinstance(payload, dict):
            raise UnsupportedResponse("Gemini 上游响应格式无效")
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
                        # Files API 下载需带 key；缺失时补上，避免结果落盘下载 401。
                        results.append(ImageResult(url=_file_uri_with_key(str(file_data["fileUri"]), api_key)))
        if not results: raise NoOutputError("Gemini 响应中没有图片")
        return results
