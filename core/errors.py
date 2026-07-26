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
