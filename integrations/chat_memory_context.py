"""ChatMemory takeover contexts 只读适配（模式参考 time_awareness）。

只通过 ``context.get_registered_star`` 使用 ChatMemory 的公开实例 API，不直接导入
对方插件模块。插件未安装、未激活、没有当前 conversation 或查询失败时均返回空列表；
ChatMemory 接管未启用时 ``build_takeover_contexts`` 返回 None，同样按空处理。
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChatMemoryContextState:
    """一次只读 takeover 查询的结果与开关状态。"""

    contexts: list[dict]
    takeover_enabled: bool = False


def _resolve_chat_memory(context: Any):
    """返回已激活且提供只读 takeover API 的 ChatMemory 实例。"""
    try:
        star = context.get_registered_star("chat_memory")
        if star is None or not bool(getattr(star, "activated", True)):
            return None
        for candidate in (
            star,
            getattr(star, "star", None),
            getattr(star, "star_cls", None),
        ):
            if candidate is not None and callable(
                getattr(candidate, "build_takeover_contexts", None)
            ):
                return candidate
    except Exception:
        return None
    return None


async def load_chat_memory_context_state(
    context: Any,
    umo: str,
    *,
    persona_id: str = "",
    user_id: str = "",
) -> ChatMemoryContextState:
    """读取与 ChatMemory 当前接管配置完全一致的 LLM ``contexts``。

    persona_id / user_id 始终传给 ChatMemory，由它自己应用 persona、cross_session、
    full_group、状态、内容白名单、前缀和预算等接管配置；查询失败降级为空历史。
    """
    chat_memory = _resolve_chat_memory(context)
    if chat_memory is None:
        return ChatMemoryContextState([])
    try:
        conversation_manager = context.conversation_manager
        conversation_id = await conversation_manager.get_curr_conversation_id(umo)
        if not conversation_id:
            return ChatMemoryContextState([])
        contexts = await chat_memory.build_takeover_contexts(
            umo=umo,
            user_id=(user_id or "").strip(),
            conversation_id=conversation_id,
            persona_id=persona_id or "",
        )
    except Exception:
        return ChatMemoryContextState([])
    if contexts is None:
        return ChatMemoryContextState([])
    return ChatMemoryContextState(list(contexts), takeover_enabled=True)
