"""事件处理阶段的参考图接管规划（纯逻辑，不依赖 AstrBot 运行时）。

明确附带在消息中的 Image 组件属于事件级临时文件，必须在事件处理阶段完成
本地化并读入内存，不能推迟到后台任务（事件结束后）才读取。本模块把这段
“接管 + 去重 + 延迟清单”逻辑抽成纯 Python，便于在不完整 mock AstrBot 的
环境下做单元测试。

对本地不可读的明确 Image 组件按来源分类处理：
- HTTP(S)：不调用 converter，进入 strict deferred，由后台
  ``fetch_reference`` 执行逐跳 SSRF 校验与大小限制，避免绕过 imago 自身的
  远程下载策略；
- ``data:``/``base64://``：事件阶段立即用 ``fetch_reference(None, ...)``
  解码接管（不需要 session），保留魔数识别与大小校验；
- 本地路径 / ``file:`` URI：直接读入内存；
- 裸文件名 / 无可用 source / 无法归类形态：事件阶段用
  ``convert_to_file_path()`` 接管。

仍无法解析时抛异常（严格失败），不静默丢图。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import unquote, urlparse
from collections.abc import Callable
from typing import Any

from .models import ImageInput
from .network import fetch_reference

_HTTP_URL = re.compile(r"https?://[^\s<>\]\[()\"']+")
_SUFFIX_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


class ReferencePlanner:
    """在事件处理阶段同步接管明确 Image 组件，生成本地参考图与延迟解析清单。

    plan() 返回 (local_references, deferred)：
    - local_references: 已读入内存的 ImageInput 列表（按内容哈希去重）；
    - deferred: 需要解析的延迟项（引用消息通常由调用方在事件阶段前台解析）：
        {"kind": "reply", "component": ..., "strict": bool}  引用消息远端回退
        {"kind": "source", "source": url, "strict": True}    明确 Image 组件的远程 HTTP(S) 参考图
        {"kind": "source", "source": url, "strict": False}   消息正文里的公网 URL

    reply 项的 strict 判定：引用正文出现图片标记，或“只有引用 id、没有可检查
    正文”的纯引用时为 True（无法排除含图，必须严格失败，避免静默丢图）；只有
    存在明确不含图的引用正文时才为 False（宽松，解析不到也不阻断）。

    明确的 Image 组件本地不可读时按来源分类：
    - HTTP(S) 进入 strict deferred（后台执行 SSRF 与大小校验，不用 converter）；
    - data:/base64:// 事件阶段立即用 fetch_reference 解码接管；
    - 本地路径/file: URI 直接读入；
    - 裸文件名/无可用 source/无法归类形态用 convert_to_file_path() 接管；
    仍无法解析时抛异常（严格失败），不会静默丢图。
    """

    def __init__(self, *, max_upload_bytes: Callable[[], int], extract_quoted_message_images=None):
        self._max_upload_bytes = max_upload_bytes
        self._extract_quoted = extract_quoted_message_images

    @staticmethod
    def _sources_of(component: Any) -> list[str]:
        sources: list[str] = []
        for field in ("path", "url", "file"):
            value = getattr(component, field, None)
            if isinstance(value, str) and value.strip():
                sources.append(value.strip())
        return sources

    @staticmethod
    def _path_of(component: Any) -> Path | None:
        for field in ("path", "url", "file"):
            value = getattr(component, field, None)
            if not isinstance(value, str) or not value.strip():
                continue
            source = value.strip()
            if source.startswith("file:"):
                parsed = urlparse(source)
                source = unquote(parsed.path or "")
                if re.match(r"^/[A-Za-z]:/", source):
                    source = source[1:]
            if source.startswith(("http://", "https://", "data:", "base64://")):
                continue
            try:
                path = Path(source).expanduser().resolve()
                if path.is_file() and not path.is_symlink():
                    return path
            except OSError:
                continue
        return None

    def _read_path(self, path: Path) -> ImageInput | None:
        try:
            path = Path(path).expanduser().resolve()
        except OSError:
            return None
        if not path.is_file() or path.is_symlink():
            return None
        data = path.read_bytes()
        if not data or len(data) > self._max_upload_bytes():
            raise ValueError("图片格式或大小不符合要求")
        mime = _SUFFIX_MIME.get(path.suffix.lower())
        if not mime:
            raise ValueError("图片格式或大小不符合要求")
        return ImageInput(data=data, mime_type=mime, filename=path.name)

    def _read_component(self, component: Any) -> ImageInput | None:
        path = self._path_of(component)
        if path is None:
            return None
        return self._read_path(path)

    @staticmethod
    async def _convert_component_to_path(component: Any) -> Path | None:
        converter = getattr(component, "convert_to_file_path", None)
        if not callable(converter):
            return None
        try:
            value = await converter()
        except Exception:
            return None
        if not value:
            return None
        try:
            return Path(str(value)).expanduser().resolve()
        except OSError:
            return None

    async def plan(self, components) -> tuple[list[ImageInput], list[dict]]:
        local_references: list[ImageInput] = []
        deferred: list[dict] = []
        seen: set[bytes] = set()

        def add_local(image: ImageInput) -> None:
            digest = hashlib.sha256(image.data).digest()
            if digest not in seen:
                seen.add(digest)
                local_references.append(image)

        def contains_image(chain) -> bool:
            for item in chain or []:
                name = type(item).__name__.lower()
                if "image" in name:
                    return True
                if name == "reply" and contains_image(getattr(item, "chain", None) or []):
                    return True
            return False

        def reply_indicates_image(component) -> bool:
            """判定 Reply 是否应按 strict 等待引用消息图片。

            返回 True 的情形：
            - 引用正文（message_str / chain 文本）出现 [图片]/[Image]/[image] 标记；
            - 只有引用 id、且没有任何可检查正文（无 chain 或 chain/message_str
              均为空）的“纯引用”——无法排除含图，必须按 strict 处理，避免后台
              静默丢图。
            返回 False 仅在存在可检查的引用正文且其中没有图片标记时（明确不含图）。
            """
            has_reply_id = any(
                str(getattr(component, field, "") or "").strip()
                for field in ("id", "message_id", "messageId")
            )
            message_str = str(getattr(component, "message_str", "") or "")
            chain_texts = [
                str(getattr(item, "text", "") or getattr(item, "content", "") or "")
                for item in getattr(component, "chain", None) or []
            ]
            markers = ("[图片]", "[Image]", "[image]")
            if any(marker in message_str for marker in markers):
                return True
            if any(marker in text for text in chain_texts for marker in markers):
                return True
            if has_reply_id and not message_str and not any(chain_texts):
                # 纯引用：有 id 但没有可检查正文（含 chain 仅空文本项），无法排除
                # 含图，按严格处理。
                return True
            return False

        async def walk(chain) -> None:
            for component in chain or []:
                name = type(component).__name__.lower()
                if name == "reply":
                    reply_chain = getattr(component, "chain", None) or []
                    has_embedded_image = contains_image(reply_chain)
                    await walk(reply_chain)
                    if not has_embedded_image and self._extract_quoted is not None:
                        deferred.append({
                            "kind": "reply",
                            "component": component,
                            "strict": reply_indicates_image(component),
                        })
                    continue
                if "image" in name:
                    image = self._read_component(component)
                    if image is not None:
                        add_local(image)
                        continue
                    sources = self._sources_of(component)
                    http_source = next(
                        (value for value in sources if value.startswith(("http://", "https://"))),
                        None,
                    )
                    if http_source is not None:
                        # 远程 URL 不在此处下载：交给后台 _resolve_deferred_references
                        # -> fetch_reference，保留 imago 自身的 SSRF 校验和大小限制。
                        deferred.append({"kind": "source", "source": http_source, "strict": True})
                        continue
                    inline_source = next(
                        (value for value in sources if value.startswith(("data:", "base64://"))),
                        None,
                    )
                    if inline_source is not None:
                        image = await fetch_reference(
                            None,
                            inline_source,
                            max_bytes=self._max_upload_bytes(),
                            block_private=False,
                        )
                        add_local(image)
                        continue
                    # 裸文件名 / 无可用 source / 无法归类形态：事件阶段用 converter 接管。
                    converted = await self._convert_component_to_path(component)
                    if converted is not None:
                        image = self._read_path(converted)
                        if image is not None:
                            add_local(image)
                            continue
                    raise ValueError("参考图无法获取")
                text = getattr(component, "text", None) or getattr(component, "content", None)
                if isinstance(text, str):
                    for url in _HTTP_URL.findall(text):
                        deferred.append({
                            "kind": "source",
                            "source": url.rstrip(".,;!?。，；！？"),
                            "strict": False,
                        })

        await walk(components)
        return local_references, deferred
