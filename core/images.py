from __future__ import annotations

import io

# 参考图上限：长边 1536、超过 600KB 才重编码。人设原图常见 5000×5000、单张 10MB+，
# 5 张 base64 后是几十 MB 的请求体，中转会截断或让模型只看到边缘（出图跑成景物）。
MAX_EDGE = 1536
REBUILD_ABOVE_BYTES = 600 * 1024
JPEG_QUALITY = 85

# 常见后缀 → MIME；未知后缀按 PNG 处理（与历史行为一致）
REFERENCE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def shrink(data: bytes, mime_type: str) -> tuple[bytes, str]:
    """把过大的参考图缩到长边 ``MAX_EDGE`` 并转 JPEG；不需要处理时原样返回。

    返回 ``(data, mime_type)``。Pillow 不可用或解码失败时原样返回，不阻塞出图。
    """
    try:
        from PIL import Image
    except ImportError:
        return data, mime_type
    try:
        # 只读文件头拿尺寸，不解码像素；尺寸和体积都合格才原样返回
        with Image.open(io.BytesIO(data)) as image:
            if max(image.size) <= MAX_EDGE and len(data) <= REBUILD_ABOVE_BYTES:
                return data, mime_type
            image = image.convert("RGB")
            image.thumbnail((MAX_EDGE, MAX_EDGE))
            buffer = io.BytesIO()
            image.save(buffer, "JPEG", quality=JPEG_QUALITY, optimize=True)
        return buffer.getvalue(), "image/jpeg"
    except Exception as exc:
        # 坏图/异常编码会静默按原图发送，只在上游报错时暴露；至少留一条可查的日志。
        try:
            from astrbot.api import logger
        except ImportError:
            logger = None
        if logger is not None:
            logger.debug("[Imago] 参考图解码失败，按原图发送 type=%s", type(exc).__name__)
        return data, mime_type


def is_animated(source) -> bool:
    """判断结果是否为动图（GIF/动态 WebP）。路径或 bytes 均可；判断失败按静态处理。"""
    try:
        from PIL import Image
    except ImportError:
        return False
    try:
        image = Image.open(io.BytesIO(source)) if isinstance(source, (bytes, bytearray)) else Image.open(source)
        with image:
            return bool(getattr(image, "is_animated", False))
    except Exception:
        return False
