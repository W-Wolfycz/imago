"""使用真实 AstrBot runtime 与 BubbleReply 验证 Imago 主动发送适配层。

普通 unittest 不自动导入本文件。它只使用假 Context，不会触发真实平台发送。
"""

import asyncio
from types import SimpleNamespace

import astrbot.api.message_components as Comp
from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
from astrbot.core.star.star_handler import EventType

from bubble_reply.main import BubbleReplyPlugin
from imago.integrations import active_send as sender_module
from imago.integrations.active_send import ProactiveSender


class _Platform:
    def meta(self):
        return SimpleNamespace(
            id="demo",
            name="demo",
            support_streaming_message=True,
        )


class _Context:
    def __init__(self, send_results=None):
        self.platform = _Platform()
        self.platform_manager = SimpleNamespace(platform_insts=[self.platform])
        self.sent = []
        self.attempted = []
        self.send_results = list(send_results or [])

    def get_platform_inst(self, platform_id):
        return self.platform if platform_id == "demo" else None

    def get_config(self, _umo=None):
        return {"provider_settings": {"streaming_response": False}}

    async def send_message(self, session, chain):
        item = (str(session), chain)
        self.attempted.append(item)
        result = self.send_results.pop(0) if self.send_results else True
        if result:
            self.sent.append(item)
        return result


class _Source:
    def __init__(self):
        message = AstrBotMessage()
        message.self_id = "bot_demo"
        message.message_id = "source_message"
        message.sender = MessageMember(user_id="10001", nickname="tester")
        self.message_obj = message
        self.plugins_name = None

    def get_sender_id(self):
        return "10001"


class _Registry:
    def __init__(self, decorating=None, after=None):
        self.decorating = [
            SimpleNamespace(handler=item, handler_full_name="decorating")
            for item in (decorating or [])
        ]
        self.after = [
            SimpleNamespace(handler=item, handler_full_name="after")
            for item in (after or [])
        ]

    def get_handlers_by_event_type(self, event_type, **_kwargs):
        if event_type == EventType.OnDecoratingResultEvent:
            return self.decorating
        if event_type == EventType.OnAfterMessageSentEvent:
            return self.after
        return []


async def _send(context, components):
    return await ProactiveSender(context, lambda *_: None).send(
        "demo:GroupMessage:group_demo",
        _Source(),
        components,
    )


async def main():
    original_registry = sender_module.star_handlers_registry

    after_calls = []

    async def after_hook(event):
        after_calls.append(event)

    plain_context = _Context()
    sender_module.star_handlers_registry = _Registry(after=[after_hook])
    try:
        plain_outcome = await _send(plain_context, [Comp.Plain("直接发送")])
    finally:
        sender_module.star_handlers_registry = original_registry
    assert plain_outcome.success is True
    assert plain_outcome.side_send_started is False
    assert len(plain_context.sent) == 1
    assert len(after_calls) == 1

    context = _Context()
    bubble = BubbleReplyPlugin(
        context,
        {
            "basic_settings": {"split_scope": "ALL"},
            "delay_settings": {"delay_seconds": 0},
        },
    )
    await bubble.initialize()
    sender_module.star_handlers_registry = _Registry(
        decorating=[bubble.on_decorating_result],
    )
    try:
        outcome = await _send(
            context,
            [Comp.At(qq="10001"), Comp.Plain("第一段\n第二段\n第三段")],
        )
    finally:
        sender_module.star_handlers_registry = original_registry
        await bubble.terminate()
    assert outcome.success is True
    assert outcome.side_send_started is True
    assert len(context.sent) == 3
    assert isinstance(context.sent[0][1].chain[0], Comp.At)

    partial_context = _Context(send_results=[False, True])
    partial_bubble = BubbleReplyPlugin(
        partial_context,
        {
            "basic_settings": {"split_scope": "ALL"},
            "delay_settings": {"delay_seconds": 0},
        },
    )
    await partial_bubble.initialize()
    sender_module.star_handlers_registry = _Registry(
        decorating=[partial_bubble.on_decorating_result],
    )
    try:
        partial_outcome = await _send(
            partial_context,
            [Comp.At(qq="10001"), Comp.Plain("第一段\n第二段\n第三段")],
        )
    finally:
        sender_module.star_handlers_registry = original_registry
        await partial_bubble.terminate()
    assert partial_outcome.success is False
    assert partial_outcome.side_send_started is True
    assert partial_outcome.side_send_error == "PlatformNotFound"
    assert len(partial_context.attempted) == 2
    assert len(partial_context.sent) == 1


if __name__ == "__main__":
    asyncio.run(main())
    print("imago active sender smoke: ok")
