from __future__ import annotations

import asyncio
import base64
import binascii
import ipaddress
import re
import socket
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

from .errors import ReferenceImageError, UnsupportedResponse
from .models import ImageInput, ImageResult

DATA_URL = re.compile(r"^data:(image/(?:png|jpeg|webp|gif));base64,(.+)$", re.I | re.S)
BASE64_URL = re.compile(r"^base64://([A-Za-z0-9+/=]*)$", re.S)

# 手动跟随重定向的最大跳数；超过即抛 ReferenceImageError。
MAX_REDIRECTS = 5
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}

_SUFFIX_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def detect_image_mime(data: bytes) -> str | None:
    """用文件魔数识别图片类型；无法识别返回 None。"""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    return None


def _max_base64_len(max_bytes: int) -> int:
    """编码 max_bytes 字节所需的 base64 字符串长度上限（4 * ceil(n / 3)）。

    用于解码前先估算：payload 长度超过该值必然解码出大于 max_bytes 的内容，
    提前拒绝可避免对超大 base64 载荷做完整解码。解码后仍保留实际长度校验。
    """
    return (max_bytes + 2) // 3 * 4


def _resolve_redirect(current: str, location: str) -> str:
    """相对/绝对 Location 统一按 RFC 3986 相对当前 URL 解析后返回。"""
    return urljoin(current, location.strip())


def _decode_base64_url(source: str, *, max_bytes: int) -> ImageInput:
    match = BASE64_URL.match(source)
    if not match or not match.group(1):
        raise ReferenceImageError("base64 图片无效")
    if len(match.group(1)) > _max_base64_len(max_bytes):
        raise ReferenceImageError("参考图过大")
    try:
        data = base64.b64decode(match.group(1), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ReferenceImageError("base64 图片无效") from exc
    if not data:
        raise ReferenceImageError("base64 图片无效")
    if len(data) > max_bytes:
        raise ReferenceImageError("参考图过大")
    mime = detect_image_mime(data)
    if not mime:
        raise ReferenceImageError("base64 图片格式无法识别")
    suffix = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}.get(mime, "png")
    return ImageInput(data=data, mime_type=mime, filename=f"base64-reference.{suffix}")


async def _resolve_checked_url(url: str, *, block_private: bool) -> tuple[str, dict | None]:
    """解析、校验并把 http 域名固定为解析出的 IP（防 DNS rebinding）。

    返回 ``(连接用 URL, 附加请求头)``：
    - URL 形式校验（HTTP(S)、无内嵌凭据）与域名解析合并完成；
    - ``block_private=True`` 时解析出的任何非公网地址直接拒绝；
    - http 域名被重写为解析出的 IP 直连并携带原 Host 头，使校验与连接使用同一
      次解析结果，消除 TOCTOU；https 保持原 URL（TLS 证书按 hostname 校验兜底，
      rebinding 需要攻击者持有合法证书，不具备现实性）。
    """
    parsed = urlparse(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError("只允许无内嵌凭据的 HTTP/HTTPS URL")
    loop = asyncio.get_running_loop()
    port = parsed.port or (80 if parsed.scheme == "http" else 443)
    try:
        infos = await loop.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError("无法解析远程主机") from exc
    addresses = {info[4][0] for info in infos}
    if block_private:
        for address in addresses:
            if not ipaddress.ip_address(address).is_global:
                raise ValueError("不允许访问私网或本地地址")
    if parsed.scheme == "http" and addresses:
        ip = next(iter(addresses))
        ip_host = f"[{ip}]" if ":" in ip else ip
        netloc = f"{ip_host}:{parsed.port}" if parsed.port else ip_host
        request_url = urlunparse(
            (parsed.scheme, netloc, parsed.path or "/", parsed.params, parsed.query, parsed.fragment)
        )
        host_name = f"[{parsed.hostname}]" if ":" in (parsed.hostname or "") else parsed.hostname
        host_header = host_name if parsed.port is None else f"{host_name}:{parsed.port}"
        return request_url, {"Host": host_header}
    return url, None


async def _fetch_http_reference(session, url: str, *, max_bytes: int, block_private: bool, verify_magic: bool) -> ImageInput:
    """HTTP(S) 参考图下载：allow_redirects=False + 手动有限跳转，逐跳做 SSRF 校验。

    - 初始 URL 与每一跳 Location 都经过 ``_resolve_checked_url``（形式校验 +
      解析 + ``block_private`` 私网/本地地址拦截，http 域名固定 IP 直连）；
      校验失败统一抛 ``ReferenceImageError``（初始 URL 保留原始原因文案，跳转
      目标用固定文案），不合法或超过跳数上限亦然。
    - 相对 Location 按当前 URL 解析后再校验，避免把内网地址藏在相对跳转里。
    - 本函数不附加任何凭据；URL 内嵌凭据被拒绝，跨主机跳转也不会携带本函数附加的头。
    - 空 body 一律拒绝；``verify_magic`` 为 True 时按文件魔数校验内容并以魔数为准
      （参考图路径），为 False 时信任 Content-Type 声明（生成结果下载保持宽松）。
    """
    current = url
    redirects = 0
    while True:
        try:
            request_url, extra_headers = await _resolve_checked_url(current, block_private=block_private)
        except ValueError as exc:
            if redirects == 0:
                raise ReferenceImageError(str(exc)) from exc
            raise ReferenceImageError("参考图重定向目标不安全") from exc
        async with session.get(request_url, allow_redirects=False, headers=extra_headers) as response:
            if response.status in _REDIRECT_STATUSES:
                redirects += 1
                if redirects > MAX_REDIRECTS:
                    raise ReferenceImageError("参考图重定向次数过多")
                location = response.headers.get("Location", "")
                if not location.strip():
                    raise ReferenceImageError("参考图重定向缺少 Location")
                current = _resolve_redirect(current, location)
                continue
            if response.status >= 400:
                raise ReferenceImageError(f"参考图 HTTP {response.status}")
            header_mime = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
            if not header_mime.startswith("image/"):
                raise ReferenceImageError("远程响应不是图片")
            data = bytearray()
            async for chunk in response.content.iter_chunked(65536):
                data.extend(chunk)
                if len(data) > max_bytes:
                    raise ReferenceImageError("参考图过大")
            if not data:
                raise ReferenceImageError("参考图无法获取")
            detected = detect_image_mime(bytes(data))
            if verify_magic:
                if not detected:
                    raise ReferenceImageError("远程响应不是图片")
                return ImageInput(data=bytes(data), mime_type=detected)
            return ImageInput(data=bytes(data), mime_type=header_mime)


async def fetch_reference(session, source: str, *, max_bytes: int, block_private: bool, verify_magic: bool = True) -> ImageInput:
    """按来源分类读取参考图（本地路径 / HTTP(S) / data: / base64://）。

    verify_magic=True（默认，参考图路径）时对 data:/HTTP(S) 内容做文件魔数识别，
    无法识别严格失败、头部声明与魔数不一致时以魔数为准；verify_magic=False 时
    信任声明/头部 MIME（生成结果下载路径保持宽松）。base64:// 无声明 MIME，
    始终按魔数识别。本地路径按扩展名映射，不受 verify_magic 影响。
    """
    match = DATA_URL.match(source)
    if match:
        if len(match.group(2)) > _max_base64_len(max_bytes):
            raise ReferenceImageError("参考图过大")
        try:
            data = base64.b64decode(match.group(2), validate=True)
        except binascii.Error as exc:
            raise ReferenceImageError("data URL 无效") from exc
        if len(data) > max_bytes:
            raise ReferenceImageError("参考图过大")
        detected = detect_image_mime(data)
        if verify_magic and not detected:
            raise ReferenceImageError("data URL 图片格式无法识别")
        if detected:
            # 头部声明的 MIME 与内容魔数不一致时以魔数为准（与 base64:// 行为一致）。
            return ImageInput(data=data, mime_type=detected)
        return ImageInput(data=data, mime_type=match.group(1).lower())
    if source.startswith("base64://"):
        return _decode_base64_url(source, max_bytes=max_bytes)
    if source.startswith(("http://", "https://")):
        return await _fetch_http_reference(session, source, max_bytes=max_bytes, block_private=block_private, verify_magic=verify_magic)
    path = Path(source).expanduser().resolve()
    if not path.is_file() or path.is_symlink(): raise ReferenceImageError("本地参考图不存在")
    data = path.read_bytes()
    if len(data) > max_bytes: raise ReferenceImageError("参考图过大")
    mime = _SUFFIX_MIME.get(path.suffix.lower())
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
    else:
        # 生成结果下载保持宽松：信任声明/头部 MIME，避免 Provider 返回非
        # png/jpeg/webp/gif 魔数的合法图片（如 AVIF）被误拒。data: 与
        # HTTP(S) URL 共用同一读取路径。
        image = await fetch_reference(session, result.url, max_bytes=max_bytes, block_private=block_private, verify_magic=False)
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
