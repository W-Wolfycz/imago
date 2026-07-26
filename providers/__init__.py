from .custom import CustomEndpointAdapter
from .dashscope import DashScopeMultimodalAdapter
from .gemini import GeminiAdapter
from .openai_chat import OpenAIChatAdapter
from .openai_image import OpenAIImageAdapter

ADAPTERS = {
    "openai_image": OpenAIImageAdapter,
    "openai_chat": OpenAIChatAdapter,
    "gemini_official": GeminiAdapter,
    "dashscope_multimodal": DashScopeMultimodalAdapter,
    "custom_endpoint": CustomEndpointAdapter,
}
