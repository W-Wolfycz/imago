from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any

from astrbot.core.message.message_event_result import MessageChain, MessageEventResult, ResultContentType
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.astrbot_message import AstrBotMessage, Group, MessageMember
from astrbot.core.platform.message_session import MessageSession
from astrbot.core.platform.message_type import MessageType
from astrbot.core.star.star_handler import EventType, star_handlers_registry


@dataclass(frozen=True)
class SendOutcome:
    success: bool
    error: str = ""
    side_effects_started: bool = False
    side_send_started: bool = False
    side_send_error: str = ""
    generation_success: bool | None = None
    usable_output_count: int | None = None
    delivery_kind: str = ""


class _ProactiveMessageEvent(AstrMessageEvent):
    """让装饰器内的 ``event.send()`` 真正委托主动发送接口。"""

    def __init__(
        self,
        *,
        context: Any,
        outbound_session: MessageSession,
        message_obj: AstrBotMessage,
        platform_meta: Any,
        session_id: str,
        before_send=None,
    ) -> None:
        self._context = context
        self._outbound_session = outbound_session
        self._before_send = before_send
        self._send_stage_started = False
        self._side_send_started = False
        self._side_send_error = ""
        super().__init__(
            message_str="",
            message_obj=message_obj,
            platform_meta=platform_meta,
            session_id=session_id,
        )

    def begin_send_stage(self) -> None:
        if self._send_stage_started:
            return
        self._send_stage_started = True
        if self._before_send:
            self._before_send()

    async def send(self, message: MessageChain) -> None:
        self.begin_send_stage()
        self._side_send_started = True
        try:
            sent = await self._context.send_message(self._outbound_session, message)
        except Exception as exc:
            self._side_send_error = type(exc).__name__
            raise
        if not sent:
            self._side_send_error = "PlatformNotFound"
            raise RuntimeError(self._side_send_error)
        await super().send(message)

    @property
    def side_send_started(self) -> bool:
        return self._side_send_started

    @property
    def side_send_error(self) -> str:
        return self._side_send_error


class ProactiveSender:
    """为主动消息补跑装饰/发送后 Hook，并使用 Context 完成实际发送。"""

    def __init__(self, context, logger, debug=None):
        self.context = context
        self.logger = logger
        self.debug = debug or (lambda *args, **kwargs: None)

    def _platform(self, session: MessageSession):
        getter = getattr(self.context, "get_platform_inst", None)
        if callable(getter):
            platform = getter(session.platform_id)
            if platform is not None:
                return platform
        manager = getattr(self.context, "platform_manager", None)
        return next(
            (item for item in getattr(manager, "platform_insts", []) if item.meta().id == session.platform_id),
            None,
        )

    def build_event(self, umo: str, source, chain: list, before_send=None) -> _ProactiveMessageEvent:
        session = MessageSession.from_str(umo)
        platform = self._platform(session)
        if platform is None:
            raise RuntimeError("找不到会话对应的平台实例")

        source_message = getattr(source, "message_obj", None)
        sender = getattr(source_message, "sender", None)
        message = AstrBotMessage()
        message.type = session.message_type
        message.session_id = session.session_id
        message.message_id = ""
        message.self_id = str(getattr(source_message, "self_id", "") or "bot")
        message.sender = MessageMember(
            user_id=str(getattr(sender, "user_id", "") or getattr(source, "get_sender_id", lambda: "")()),
            nickname=getattr(sender, "nickname", None),
        )
        message.group = Group(group_id=session.session_id) if session.message_type == MessageType.GROUP_MESSAGE else None
        message.message = []
        message.message_str = ""
        message.raw_message = None

        event = _ProactiveMessageEvent(
            context=self.context,
            outbound_session=session,
            message_obj=message,
            platform_meta=platform.meta(),
            session_id=session.session_id,
            before_send=before_send,
        )
        event.plugins_name = getattr(source, "plugins_name", None)
        result = MessageEventResult(chain=list(chain))
        result.set_result_content_type(ResultContentType.LLM_RESULT)
        event.set_result(result)
        return event

    async def _run_hooks(self, event, event_type) -> None:
        handlers = star_handlers_registry.get_handlers_by_event_type(
            event_type,
            plugins_name=getattr(event, "plugins_name", None),
        )
        self.debug("[Imago] 主动发送 Hook event=%s handlers=%d", event_type.name, len(handlers))
        for handler in handlers:
            if event.is_stopped():
                break
            callback = getattr(handler, "handler", handler)
            try:
                value = callback(event)
                if inspect.isawaitable(value):
                    await value
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                self.logger(f"Hook 执行异常 event={event_type.name}", exc)

    async def send(self, umo: str, source, chain: list, before_send=None) -> SendOutcome:
        try:
            event = self.build_event(umo, source, chain, before_send=before_send)
        except Exception as exc:
            return SendOutcome(False, type(exc).__name__, False)

        await self._run_hooks(event, EventType.OnDecoratingResultEvent)
        if event.is_stopped():
            return SendOutcome(
                not bool(event.side_send_error),
                event.side_send_error,
                side_effects_started=True,
                side_send_started=event.side_send_started,
                side_send_error=event.side_send_error,
            )
        result = event.get_result()
        processed = list(getattr(result, "chain", None) or [])
        self.debug("[Imago] 主动发送装饰完成 components=%d stopped=%s", len(processed), event.is_stopped())
        error = ""
        success = True
        if processed:
            event.begin_send_stage()
            try:
                success = bool(await self.context.send_message(umo, MessageChain(chain=processed)))
                if not success:
                    error = "PlatformNotFound"
            except Exception as exc:
                success = False
                error = type(exc).__name__
                self.logger("主动消息发送异常", exc)
        await self._run_hooks(event, EventType.OnAfterMessageSentEvent)
        if event.side_send_error:
            success = False
            error = error or f"SideSend:{event.side_send_error}"
        self.debug("[Imago] 主动发送结束 success=%s error=%s", success, error or "none")
        return SendOutcome(
            success,
            error,
            side_effects_started=True,
            side_send_started=event.side_send_started,
            side_send_error=event.side_send_error,
        )
