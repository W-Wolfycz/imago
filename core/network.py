from __future__ import annotations

import base64
import binascii
import re
from pathlib import Path

from .errors import ReferenceImageError, UnsupportedResponse
from .models import ImageInput, ImageResult
from .security import validate_remote_url

DATA_URL = re.compile(r"^data:(image/(?:png|jpeg|webp|gif));base64,(.+)$", re.I | re.S)


async def fetch_reference(session, source: str, *, max_bytes: int, block_private: bool) -> ImageInput:
    match = DATA_URL.match(source)
    if match:
        try: data = base64.b64decode(match.group(2), validate=True)
        except binascii.Error as exc: raise ReferenceImageError("data URL 无效") from exc
        if len(data) > max_bytes: raise ReferenceImageError("参考图过大")
        return ImageInput(data=data, mime_type=match.group(1).lower())
    if source.startswith(("http://", "https://")):
        validate_remote_url(source, block_private=block_private)
        async with session.get(source) as response:
            if response.status >= 400: raise ReferenceImageError(f"参考图 HTTP {response.status}")
            mime = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
            if not mime.startswith("image/"): raise ReferenceImageError("远程响应不是图片")
            data = bytearray()
            async for chunk in response.content.iter_chunked(65536):
                data.extend(chunk)
                if len(data) > max_bytes: raise ReferenceImageError("参考图过大")
            return ImageInput(data=bytes(data), mime_type=mime)
    path = Path(source).expanduser().resolve()
    if not path.is_file() or path.is_symlink(): raise ReferenceImageError("本地参考图不存在")
    data = path.read_bytes()
    if len(data) > max_bytes: raise ReferenceImageError("参考图过大")
    suffix_mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}
    mime = suffix_mime.get(path.suffix.lower())
    if not mime: raise ReferenceImageError("本地参考图格式不支持")
    return ImageInput(data=data, mime_type=mime, filename=path.name)


async def materialize_result(session, result: ImageResult, directory: Path, *, max_bytes: int, block_private: bool) -> Path | str:
    if result.local_path:
        path = result.local_path.resolve()
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            raise UnsupportedResponse("本地生成结果不可用")
        return path
    if result.data is not None:
        data, mime = result.data, result.mime_type
    elif DATA_URL.match(result.url):
        image = await fetch_reference(session, result.url, max_bytes=max_bytes, block_private=block_private)
        data, mime = image.data, image.mime_type
    else:
        image = await fetch_reference(session, result.url, max_bytes=max_bytes, block_private=block_private)
        data, mime = image.data, image.mime_type
    if not data:
        raise UnsupportedResponse("生成结果为空")
    if len(data) > max_bytes:
        raise UnsupportedResponse("生成结果超过存储上限")
    directory.mkdir(parents=True, exist_ok=True)
    suffix = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}.get(mime, ".png")
    import uuid
    path = directory / f"{uuid.uuid4().hex}{suffix}"
    path.write_bytes(data)
    return path
