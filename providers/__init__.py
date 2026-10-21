from .custom import CustomEndpointAdapter
from .gemini import GeminiAdapter
from .openai_image import OpenAIImageAdapter
from .xai_image import XaiImageAdapter

ADAPTERS = {
    "openai_image": OpenAIImageAdapter,
    "xai_image": XaiImageAdapter,
    "gemini_official": GeminiAdapter,
    "custom_endpoint": CustomEndpointAdapter,
}
