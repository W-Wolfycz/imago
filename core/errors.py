class ImagoError(Exception):
    code = "internal_error"


class ConfigurationError(ImagoError):
    code = "configuration_missing"


class PersonaError(ImagoError):
    code = "persona_invalid"


class ReferenceImageError(ImagoError):
    code = "reference_invalid"


class QuotaError(ImagoError):
    code = "quota_denied"


class ProviderError(ImagoError):
    code = "provider_error"

    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


class ProviderTimeout(ProviderError):
    code = "provider_timeout"


class UnsupportedResponse(ProviderError):
    code = "unsupported_response"


class NoOutputError(UnsupportedResponse):
    """Provider 请求已完成，但响应中没有可用图片。"""

    code = "no_output"


class SendError(ImagoError):
    code = "send_failed"


class DuplicateImage(ImagoError):
    code = "duplicate_image"


# 创建阶段允许透出的固定文案：前缀白名单（冒号结尾的条目允许后跟数字等参数，
# 如 "绘图额度不足: 3"、"不允许的附加参数: n"），不带冒号的只允许完全一致。
_ALLOWED_ERROR_MESSAGES = (
    "提示词不能为空",
    "未配置有效图片节点",
    "附加参数必须使用 --key value",
    "不允许的附加参数:",
    "当前 AstrBot 不支持 Persona 正式解析",
    "Persona 不存在或 prompt 为空",
    "任务准备超时",
    "引用消息图片无法获取",
    "插件正在关闭",
    "你当前无法使用绘图功能",
    "绘图额度不足:",
    "绘图额度不足：",
    "无法识别用户 ID",
    "用户 ID 无效",
    "额度不能小于 0",
    "请提供用户 ID 和额度整数",
    "额度必须是整数",
    "外观摘要不能为空",
    "请在同一条消息中附带图片",
)

# 参考图/SSRF 相关的固定文案：必须与已知文案完全一致才透出，
# 防止拼接了 URL/路径/平台信息的变体被前缀误放行。
_REFERENCE_ERROR_MESSAGES = frozenset((
    "参考图重定向次数过多",
    "参考图重定向目标不安全",
    "参考图重定向缺少 Location",
    "参考图过大",
    "参考图无法获取",
    "远程响应不是图片",
    "base64 图片无效",
    "base64 图片格式无法识别",
    "data URL 无效",
    "data URL 图片格式无法识别",
    "本地参考图不存在",
    "本地参考图格式不支持",
    "图片格式或大小不符合要求",
    "不允许访问私网或本地地址",
    "无法解析远程主机",
    "只允许无内嵌凭据的 HTTP/HTTPS URL",
))


def safe_creation_error_message(exc: BaseException) -> str:
    """只向用户和主 LLM 返回任务创建阶段的可公开原因。"""
    from .security import redact

    message = redact(str(exc)).strip()
    for prefix in _ALLOWED_ERROR_MESSAGES:
        if message == prefix or (
            prefix.endswith((":", "：")) and message.startswith(prefix)
        ):
            return message
    if message in _REFERENCE_ERROR_MESSAGES:
        return message
    # “参考图 HTTP <3 位状态码>” 只含状态码，不包含 URL/路径/平台信息。
    if message.startswith("参考图 HTTP "):
        status = message[len("参考图 HTTP "):]
        if len(status) == 3 and status.isdigit():
            return message
    if isinstance(exc, (TypeError, ValueError, OverflowError)):
        return "任务参数无效"
    return "插件暂时无法创建任务"
