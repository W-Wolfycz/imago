"""Imago 插件类入口级行为测试。

不导入真实 `astrbot.core`：安装轻量 stub（模式同 wakelite/test/test_wakelite.py），
只覆盖插件自身逻辑——命令/工具入口、人设解析数据源、`_finish_task` 终态链、
主动发送、ChatMemory 只读接管、Provider adapter 请求体/响应解析。历史上
chat_memory 曾因 notice/request 低频分支炸 AttributeError，入口级覆盖就是为
堵住这一类回归。
"""

from __future__ import annotations

import asyncio
import base64
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]


def _install_astrbot_stubs() -> None:
    if "astrbot.api" in sys.modules:
        return

    class _Logger:
        def debug(self, *a, **k): pass
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def error(self, *a, **k): pass

    class Plain:
        def __init__(self, text=""):
            self.text = text

    class Reply:
        def __init__(self, id="", sender_id=""):
            self.id = id
            self.sender_id = sender_id

    class At:
        def __init__(self, qq=""):
            self.qq = qq

    class Image:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class MessageChain(list):
        def __init__(self, chain=None, **kwargs):
            super().__init__(chain or [])

    class MessageEventResult:
        def __init__(self, chain=None):
            self.chain = chain or []

        def set_result_content_type(self, content_type):
            self.content_type = content_type

    class ResultContentType:
        LLM_RESULT = "llm_result"

    class ProviderRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class AstrMessageEvent:
        def __init__(self, message_str="", message_obj=None, platform_meta=None, session_id=""):
            self.message_str = message_str
            self.message_obj = message_obj
            self.platform_meta = platform_meta
            self.session_id = session_id
            self.plugins_name = None
            self._result = None
            self._stopped = False

        def get_result(self):
            return self._result

        def set_result(self, result):
            self._result = result

        def clear_result(self):
            self._result = None

        def is_stopped(self):
            return self._stopped

        def stop_event(self):
            self._stopped = True

        def begin_send_stage(self):
            pass

        async def send(self, message):
            pass

        def get_sender_id(self):
            return "10001"

        def get_platform_id(self):
            return "test"

        def get_platform_name(self):
            return "test"

        def get_messages(self):
            return []

    class MessageSession:
        def __init__(self, platform_id="", message_type="", session_id=""):
            self.platform_id = platform_id
            self.message_type = message_type
            self.session_id = session_id

        @classmethod
        def from_str(cls, value):
            parts = str(value).split(":", 2)
            return cls(
                parts[0] if len(parts) > 0 else "",
                parts[1] if len(parts) > 1 else "",
                parts[2] if len(parts) > 2 else "",
            )

    class MessageType:
        GROUP_MESSAGE = "group"
        FRIEND_MESSAGE = "friend"

    class MessageMember:
        def __init__(self, user_id="", nickname=None):
            self.user_id = user_id
            self.nickname = nickname

    class Group:
        def __init__(self, group_id=""):
            self.group_id = group_id

    class AstrBotMessage:
        def __init__(self):
            self.type = None
            self.session_id = ""
            self.message_id = ""
            self.self_id = ""
            self.sender = None
            self.group = None
            self.message = []
            self.message_str = ""
            self.raw_message = None

    class _CommandGroup:
        def __init__(self, name):
            self.name = name

        def __call__(self, func):
            # 用作 group 装饰器（如 @filter.command_group("imago")）时返回自身，
            # 保证类体内可继续 .group()/.command() 链式声明。
            self.func = func
            return self

        def group(self, name):
            return _CommandGroup(name)

        def command(self, name):
            # 叶子命令：原样返回被装饰函数，保证实例属性查找拿到协程函数。
            return lambda func: func

    class Filter:
        @staticmethod
        def command_group(name):
            return _CommandGroup(name)

        @staticmethod
        def command(name):
            return lambda func: func

        @staticmethod
        def llm_tool(name):
            return lambda func: func

        @staticmethod
        def on_llm_request(priority=0):
            return lambda func: func

        @staticmethod
        def permission_type(*a, **k):
            return lambda func: func

        class PermissionType:
            ADMIN = "admin"
            ALL = "all"

        @staticmethod
        def event_message_type(*a, **k):
            return lambda func: func

        @staticmethod
        def platform_adapter_type(*a, **k):
            return lambda func: func

        @staticmethod
        def on_decorating_result(*a, **k):
            return lambda func: func

        @staticmethod
        def on_after_message_sent(*a, **k):
            return lambda func: func

        class EventMessageType:
            ALL = "all"
            GROUP_MESSAGE = "group"

        class PlatformAdapterType:
            AIOCQHTTP = "aiocqhttp"

    class Star:
        def __init__(self, context):
            self.context = context

    class Context:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def get_config(self, umo=None):
            raise NotImplementedError

        def get_registered_star(self, name):
            return None

    class StarTools:
        @staticmethod
        def get_data_dir(plugin_name=None):
            return tempfile.gettempdir()

    def register(name, author, desc, version):
        def decorator(cls):
            return cls

        return decorator

    class EventType:
        OnDecoratingResultEvent = SimpleNamespace(name="on_decorating_result")
        OnAfterMessageSentEvent = SimpleNamespace(name="after_message_sent")

    class _HandlerRegistry:
        def __init__(self):
            self.handlers = {}

        def get_handlers_by_event_type(self, event_type, plugins_name=None):
            return list(self.handlers.get(event_type, []))

    star_handlers_registry = _HandlerRegistry()

    class ClientTimeout:
        def __init__(self, total=None, connect=None):
            pass

    class ClientSession:
        def __init__(self, timeout=None, **kwargs):
            pass

    class FormData:
        def __init__(self):
            self._fields = []

        def add_field(self, name, value, filename=None, content_type=None):
            self._fields.append((name, value))

    aiohttp = types.ModuleType("aiohttp")
    aiohttp.ClientTimeout = ClientTimeout
    aiohttp.ClientSession = ClientSession
    aiohttp.FormData = FormData

    quart = types.ModuleType("quart")
    quart.jsonify = lambda *a, **k: {"quart_json": (a, k)}
    quart.request = SimpleNamespace(args={})

    modules = {
        "astrbot": types.ModuleType("astrbot"),
        "astrbot.api": types.ModuleType("astrbot.api"),
        "astrbot.api.event": types.ModuleType("astrbot.api.event"),
        "astrbot.api.message_components": types.ModuleType("astrbot.api.message_components"),
        "astrbot.api.provider": types.ModuleType("astrbot.api.provider"),
        "astrbot.api.star": types.ModuleType("astrbot.api.star"),
        "astrbot.core": types.ModuleType("astrbot.core"),
        "astrbot.core.message": types.ModuleType("astrbot.core.message"),
        "astrbot.core.message.message_event_result": types.ModuleType("astrbot.core.message.message_event_result"),
        "astrbot.core.platform": types.ModuleType("astrbot.core.platform"),
        "astrbot.core.platform.astr_message_event": types.ModuleType("astrbot.core.platform.astr_message_event"),
        "astrbot.core.platform.astrbot_message": types.ModuleType("astrbot.core.platform.astrbot_message"),
        "astrbot.core.platform.message_session": types.ModuleType("astrbot.core.platform.message_session"),
        "astrbot.core.platform.message_type": types.ModuleType("astrbot.core.platform.message_type"),
        "astrbot.core.star": types.ModuleType("astrbot.core.star"),
        "astrbot.core.star.star_handler": types.ModuleType("astrbot.core.star.star_handler"),
        "aiohttp": aiohttp,
        "quart": quart,
    }
    sys.modules.update(modules)

    modules["astrbot.api"].logger = _Logger()
    modules["astrbot.api"].Star = Star
    modules["astrbot.api"].Context = Context
    modules["astrbot.api"].StarTools = StarTools
    modules["astrbot.api"].register = register
    modules["astrbot.api.star"].Star = Star
    modules["astrbot.api.star"].Context = Context
    modules["astrbot.api.star"].StarTools = StarTools
    modules["astrbot.api.star"].register = register
    modules["astrbot.api.event"].AstrMessageEvent = AstrMessageEvent
    modules["astrbot.api.event"].filter = Filter
    modules["astrbot.api.message_components"].At = At
    modules["astrbot.api.message_components"].Image = Image
    modules["astrbot.api.message_components"].Plain = Plain
    modules["astrbot.api.message_components"].Reply = Reply
    modules["astrbot.api.provider"].ProviderRequest = ProviderRequest
    modules["astrbot.core.message.message_event_result"].MessageChain = MessageChain
    modules["astrbot.core.message.message_event_result"].MessageEventResult = MessageEventResult
    modules["astrbot.core.message.message_event_result"].ResultContentType = ResultContentType
    modules["astrbot.core.platform.astr_message_event"].AstrMessageEvent = AstrMessageEvent
    modules["astrbot.core.platform.astrbot_message"].AstrBotMessage = AstrBotMessage
    modules["astrbot.core.platform.astrbot_message"].MessageMember = MessageMember
    modules["astrbot.core.platform.astrbot_message"].Group = Group
    modules["astrbot.core.platform.message_session"].MessageSession = MessageSession
    modules["astrbot.core.platform.message_type"].MessageType = MessageType
    modules["astrbot.core.star.star_handler"].EventType = EventType
    modules["astrbot.core.star.star_handler"].star_handlers_registry = star_handlers_registry


_install_astrbot_stubs()

from imago import main as main_module  # noqa: E402
from imago.integrations import active_send, chat_memory_context  # noqa: E402
from imago.core.models import DrawTask, GenerationRequest, ImageInput, ImageResult, TaskState  # noqa: E402
from imago.providers.custom import CustomEndpointAdapter  # noqa: E402
from imago.providers.gemini import GeminiAdapter, _file_uri_with_key  # noqa: E402
from imago.providers.openai_chat import OpenAIChatAdapter  # noqa: E402

from astrbot.api.event import AstrMessageEvent as _BaseEvent  # noqa: E402


class FakeEvent(_BaseEvent):
    """命令/工具处理链用的最小事件。"""

    def __init__(self, umo="test:group:group_demo"):
        super().__init__()
        self.unified_msg_origin = umo
        self.message_obj = SimpleNamespace(message_id=None, self_id="bot")

    def plain_result(self, text):
        from astrbot.api.message_components import Plain
        from astrbot.core.message.message_event_result import MessageChain

        result = SimpleNamespace(chain=MessageChain([Plain(text)]))
        return result


def _make_plugin(testcase: unittest.TestCase, raw_config=None, context=None):
    """构造 Imago 实例：数据目录指向独立临时目录，避免测试互相污染。"""
    tmp = tempfile.mkdtemp(prefix="imago-entry-test-")
    patcher = patch.object(main_module.StarTools, "get_data_dir", return_value=tmp)
    patcher.start()
    testcase.addCleanup(patcher.stop)
    plugin = main_module.Imago(context or SimpleNamespace(get_config=lambda umo=None: {}), config=raw_config or {})
    return plugin


class SafeCreationErrorTests(unittest.TestCase):
    def test_exact_whitelist_messages_pass_through(self):
        for text in (
            "提示词不能为空",
            "未配置有效图片节点",
            "Persona 不存在或 prompt 为空",
            "引用消息图片无法获取",
            "插件正在关闭",
            "外观摘要不能为空",
            "请在同一条消息中附带图片",
        ):
            with self.subTest(text=text):
                self.assertEqual(main_module.Imago._safe_creation_error(ValueError(text)), text)

    def test_colon_prefix_whitelist_allows_parameters(self):
        self.assertEqual(
            main_module.Imago._safe_creation_error(ValueError("绘图额度不足: 3")),
            "绘图额度不足: 3",
        )
        self.assertEqual(
            main_module.Imago._safe_creation_error(ValueError("不允许的附加参数: n")),
            "不允许的附加参数: n",
        )

    def test_reference_fixed_messages_pass_through_exactly(self):
        for text in ("参考图过大", "远程响应不是图片", "不允许访问私网或本地地址", "图片格式或大小不符合要求"):
            with self.subTest(text=text):
                self.assertEqual(main_module.Imago._safe_creation_error(ValueError(text)), text)
        # 拼接变体不得放行
        self.assertEqual(
            main_module.Imago._safe_creation_error(ValueError("参考图过大 extra junk")),
            "任务参数无效",
        )

    def test_http_status_only_exactly_three_digits(self):
        self.assertEqual(main_module.Imago._safe_creation_error(ValueError("参考图 HTTP 502")), "参考图 HTTP 502")
        self.assertEqual(main_module.Imago._safe_creation_error(ValueError("参考图 HTTP 50")), "任务参数无效")
        self.assertEqual(main_module.Imago._safe_creation_error(ValueError("参考图 HTTP 5020")), "任务参数无效")

    def test_unknown_errors_are_generic_and_redacted(self):
        self.assertEqual(
            main_module.Imago._safe_creation_error(RuntimeError("boom")),
            "插件暂时无法创建任务",
        )
        message = main_module.Imago._safe_creation_error(RuntimeError("api_key=sk-verysecret"))
        self.assertNotIn("sk-verysecret", message)


class ToolHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.plugin = _make_plugin(self, {"providers": [{"id": "node", "api_type": "openai_image", "base_url": "https://example.invalid", "api_keys": ["k"], "model": "m"}]})
        self.plugin._submit = AsyncMock()

    def _event(self):
        return FakeEvent()

    async def test_persona_tool_normal_success_wording(self):
        result = await self.plugin.generate_persona_image(self._event(), "自拍")
        self.assertIn("后台 Persona 图片任务已创建", result)
        self.assertIn("正在拍摄，请稍后", result)

    async def test_persona_tool_creation_failure_whitelisted(self):
        plugin = _make_plugin(self, {})
        plugin._submit = AsyncMock(side_effect=ValueError("未配置有效图片节点"))
        result = await plugin.generate_persona_image(self._event(), "自拍")
        self.assertIn("未能创建", result)
        self.assertIn("未配置有效图片节点", result)

    async def test_image_tool_normal_success_wording(self):
        result = await self.plugin.generate_image(self._event(), "画一只猫")
        self.assertIn("后台绘图任务已创建", result)
        self.assertIn("正在绘制", result)

    async def test_image_tool_creation_failure_whitelisted(self):
        plugin = _make_plugin(self, {})
        plugin._submit = AsyncMock(side_effect=ValueError("提示词不能为空"))
        result = await plugin.generate_image(self._event(), "")
        self.assertIn("未能创建", result)
        self.assertIn("提示词不能为空", result)


class SummaryCommandTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.plugin = _make_plugin(self)
        self.plugin._resolve_persona = AsyncMock(return_value=("persona_demo", "prompt-demo"))

    async def test_summary_rebuild_executes_once_and_yields_result(self):
        calls = []

        async def tracked(persona_id):
            calls.append(persona_id)
            return "新摘要"

        self.plugin.rebuild_summary = tracked
        results = [item async for item in self.plugin.summary_rebuild(FakeEvent())]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], "persona_demo")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chain[0].text, "新摘要")

    async def test_summary_show_error_uses_whitelist(self):
        self.plugin._resolve_persona = AsyncMock(side_effect=ValueError("Persona 不存在或 prompt 为空"))
        results = [item async for item in self.plugin.summary_show(FakeEvent())]
        self.assertEqual(results[0].chain[0].text, "Persona 不存在或 prompt 为空")

    async def test_summary_set_error_uses_whitelist(self):
        self.plugin.store.set_summary = lambda *a, **k: (_ for _ in ()).throw(ValueError("外观摘要不能为空"))
        results = [item async for item in self.plugin.summary_set(FakeEvent(), "")]
        self.assertEqual(results[0].chain[0].text, "外观摘要不能为空")

    async def test_summary_rebuild_error_uses_whitelist(self):
        self.plugin._resolve_persona = AsyncMock(side_effect=ValueError("Persona 不存在或 prompt 为空"))
        results = [item async for item in self.plugin.summary_rebuild(FakeEvent())]
        self.assertEqual(results[0].chain[0].text, "Persona 不存在或 prompt 为空")


class ResolvePersonaTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_settings_prefer_umo_config(self):
        captured = {}

        class PersonaManager:
            async def resolve_selected_persona(self, **kwargs):
                captured.update(kwargs)
                return "gpt_demo", {"prompt": "prompt " * 20}, None, False

        class ConversationManager:
            async def get_curr_conversation_id(self, umo):
                return None

        context = SimpleNamespace(
            persona_manager=PersonaManager(),
            conversation_manager=ConversationManager(),
        )
        context.get_config = lambda umo=None: (
            {"provider_settings": {"default_personality": "gpt_demo"}}
            if umo is not None
            else {"provider_settings": {"default_personality": "persona_demo"}}
        )
        plugin = _make_plugin(self, context=context)
        persona_id, prompt = await plugin._resolve_persona(FakeEvent(umo="test:group:group_demo"))
        self.assertEqual(persona_id, "gpt_demo")
        self.assertEqual(captured["provider_settings"], {"default_personality": "gpt_demo"})
        self.assertTrue(prompt.strip())

    async def test_provider_settings_fallback_to_default_when_umo_config_missing(self):
        captured = {}

        class PersonaManager:
            async def resolve_selected_persona(self, **kwargs):
                captured.update(kwargs)
                return "persona_demo", {"prompt": "prompt " * 20}, None, False

        class ConversationManager:
            async def get_curr_conversation_id(self, umo):
                return None

        context = SimpleNamespace(
            persona_manager=PersonaManager(),
            conversation_manager=ConversationManager(),
        )
        context.get_config = lambda umo=None: (
            None if umo is not None else {"provider_settings": {"default_personality": "persona_demo"}}
        )
        plugin = _make_plugin(self, context=context)
        persona_id, _ = await plugin._resolve_persona(FakeEvent())
        self.assertEqual(persona_id, "persona_demo")
        self.assertEqual(captured["provider_settings"], {"default_personality": "persona_demo"})


class FinishTaskTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.sent = []
        self.plugin = _make_plugin(self)
        self.plugin.scheduler = SimpleNamespace(set_stage=lambda task, stage: None, accepting=True, tasks=set())
        self.plugin.sender = SimpleNamespace(
            send=AsyncMock(side_effect=self._record_send)
        )

    async def _record_send(self, umo, event, chain, before_send=None):
        self.sent.append({"umo": umo, "chain": list(chain)})
        from imago.integrations.active_send import SendOutcome

        return SendOutcome(True, "", side_effects_started=True, side_send_started=True)

    def _task(self, state):
        task = DrawTask("a1b2c3d4e5f60718", "test:group:group_demo", GenerationRequest("画图"))
        task.state = state
        task.runtime["source_event"] = FakeEvent()
        return task

    async def test_timeout_chain_says_timeout(self):
        task = self._task(TaskState.TIMED_OUT)
        await self.plugin._finish_task(task, [])
        texts = [item.text for item in self.sent[0]["chain"] if hasattr(item, "text")]
        self.assertEqual(texts[-1], "绘制超时，请稍后再试。")

    async def test_failed_chain_says_failed(self):
        task = self._task(TaskState.FAILED)
        await self.plugin._finish_task(task, [])
        texts = [item.text for item in self.sent[0]["chain"] if hasattr(item, "text")]
        self.assertEqual(texts[-1], "绘制失败，请稍后再试。")

    async def test_normal_delivery_records_success_flags(self):
        task = self._task(TaskState.FAILED)
        await self.plugin._finish_task(task, [])
        self.assertTrue(task.runtime["notification_delivery_attempted"])
        self.assertTrue(task.runtime["notification_delivery_success"])
        self.assertTrue(task.runtime["runner_send_completed"])


class RemoveGenerationToolsTests(unittest.TestCase):
    def test_blocked_access_removes_both_tools(self):
        removed = []

        class ToolSet:
            def remove_tool(self, name):
                removed.append(name)

        plugin = _make_plugin(self)
        plugin._quota_access = lambda event, amount=1: SimpleNamespace(allowed=False)
        req = SimpleNamespace(func_tool=ToolSet())
        plugin._remove_unavailable_generation_tools(FakeEvent(), req)
        self.assertEqual(removed, ["generate_image", "generate_persona_image"])

    def test_allowed_access_keeps_tools(self):
        removed = []

        class ToolSet:
            def remove_tool(self, name):
                removed.append(name)

        plugin = _make_plugin(self)
        plugin._quota_access = lambda event, amount=1: SimpleNamespace(allowed=True)
        req = SimpleNamespace(func_tool=ToolSet())
        plugin._remove_unavailable_generation_tools(FakeEvent(), req)
        self.assertEqual(removed, [])


class ProactiveSenderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember

        self.sent_messages = []
        self.context = SimpleNamespace()
        self.context.send_message = AsyncMock(side_effect=self._fake_send_message)
        self.platform = SimpleNamespace(meta=lambda: SimpleNamespace(id="test"))
        self.context.get_platform_inst = lambda platform_id: self.platform
        self.context.platform_manager = None
        self.logger = lambda message, exc: None
        self.sender = active_send.ProactiveSender(self.context, self.logger, lambda *a, **k: None)

        message = AstrBotMessage()
        message.sender = MessageMember(user_id="10001", nickname="user_demo")
        self.source = FakeEvent()
        self.source.message_obj = message
        self.registry_patcher = patch.object(
            active_send.star_handlers_registry,
            "get_handlers_by_event_type",
            side_effect=self._handlers_for,
        )
        self.registry_patcher.start()
        self.addCleanup(self.registry_patcher.stop)
        self._decorating = []
        self._after = []

    async def _fake_send_message(self, session, chain):
        self.sent_messages.append(list(chain))
        return True

    def _handlers_for(self, event_type, plugins_name=None):
        from astrbot.core.star.star_handler import EventType

        if event_type == EventType.OnDecoratingResultEvent:
            return list(self._decorating)
        if event_type == EventType.OnAfterMessageSentEvent:
            return list(self._after)
        return []

    def _chain(self):
        from astrbot.api.message_components import Plain

        return [Plain("结果")]

    async def test_normal_send_delivers_chain(self):
        outcome = await self.sender.send("test:group:group_demo", self.source, self._chain())
        self.assertTrue(outcome.success)
        self.assertEqual(len(self.sent_messages), 1)
        self.assertEqual(self.sent_messages[0][0].text, "结果")

    async def test_empty_chain_after_decoration_is_delivery_failure(self):
        def clear(event):
            event.clear_result()

        self._decorating = [SimpleNamespace(handler=clear)]
        outcome = await self.sender.send("test:group:group_demo", self.source, self._chain())
        self.assertFalse(outcome.success)
        self.assertEqual(outcome.error, "DecoratorSuppressed")
        self.assertEqual(self.sent_messages, [])

    async def test_stopped_event_reports_side_send_state(self):
        def stop(event):
            event.stop_event()

        self._decorating = [SimpleNamespace(handler=stop)]
        outcome = await self.sender.send("test:group:group_demo", self.source, self._chain())
        self.assertTrue(outcome.success)
        self.assertTrue(outcome.side_effects_started)

    async def test_before_send_runs_on_begin_send_stage(self):
        marker = []
        outcome = await self.sender.send(
            "test:group:group_demo",
            self.source,
            self._chain(),
            before_send=lambda: marker.append(True),
        )
        self.assertTrue(outcome.success)
        self.assertEqual(marker, [True])


class ChatMemoryContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_star_returns_empty(self):
        context = SimpleNamespace(get_registered_star=lambda name: None)
        state = await chat_memory_context.load_chat_memory_context_state(context, "test:group:group_demo")
        self.assertEqual(state.contexts, [])
        self.assertFalse(state.takeover_enabled)

    async def test_takeover_contexts_returned(self):
        class FakeCM:
            async def build_takeover_contexts(self, **kwargs):
                return [{"role": "user", "content": "历史"}]

        star = SimpleNamespace(activated=True, star=None, star_cls=FakeCM())
        conversation_manager = SimpleNamespace(get_curr_conversation_id=AsyncMock(return_value="conv_demo"))
        context = SimpleNamespace(
            get_registered_star=lambda name: star,
            conversation_manager=conversation_manager,
        )
        state = await chat_memory_context.load_chat_memory_context_state(
            context, "test:group:group_demo", persona_id="persona_demo", user_id="10001"
        )
        self.assertTrue(state.takeover_enabled)
        self.assertEqual(state.contexts, [{"role": "user", "content": "历史"}])

    async def test_no_conversation_returns_empty(self):
        class FakeCM:
            async def build_takeover_contexts(self, **kwargs):
                return [{"role": "user", "content": "历史"}]

        star = SimpleNamespace(activated=True, star=None, star_cls=FakeCM())
        conversation_manager = SimpleNamespace(get_curr_conversation_id=AsyncMock(return_value=None))
        context = SimpleNamespace(
            get_registered_star=lambda name: star,
            conversation_manager=conversation_manager,
        )
        state = await chat_memory_context.load_chat_memory_context_state(context, "test:group:group_demo")
        self.assertEqual(state.contexts, [])

    async def test_exception_degrades_to_empty(self):
        class FakeCM:
            async def build_takeover_contexts(self, **kwargs):
                raise RuntimeError("boom")

        star = SimpleNamespace(activated=True, star=None, star_cls=FakeCM())
        conversation_manager = SimpleNamespace(get_curr_conversation_id=AsyncMock(return_value="conv_demo"))
        context = SimpleNamespace(
            get_registered_star=lambda name: star,
            conversation_manager=conversation_manager,
        )
        state = await chat_memory_context.load_chat_memory_context_state(context, "test:group:group_demo")
        self.assertEqual(state.contexts, [])


class _FakeResponse:
    def __init__(self, status=200, payload=None):
        self.status = status
        self._payload = payload

    async def json(self, content_type=None):
        return self._payload


class _FakeSession:
    def __init__(self, response=None):
        self.response = response or _FakeResponse()
        self.calls = []

    def post(self, url, headers=None, json=None, data=None, **kwargs):
        self.calls.append({"url": url, "headers": headers, "json": json, "data": data})

        class _CM:
            async def __aenter__(self):
                return self.response

            async def __aexit__(self, *args):
                return False

        _CM.response = self.response
        return _CM()


class ProviderAdapterTests(unittest.IsolatedAsyncioTestCase):
    def _request(self, **kwargs):
        return GenerationRequest("画一只猫", **kwargs)

    async def test_custom_endpoint_body_and_parse(self):
        from imago.core.models import ProviderConfig

        session = _FakeSession(_FakeResponse(200, {"data": [{"url": "https://example.invalid/a.png"}]}))
        adapter = CustomEndpointAdapter(ProviderConfig("node", "custom_endpoint", "https://example.invalid", ("key_demo",), model="m"))
        results = await adapter.generate(
            session,
            self._request(size="1024x1024", aspect_ratio="1:1", extra_params={"quality": "high"},
                          references=[ImageInput(b"imgdata", "image/png", "r.png")]),
            "key_demo",
        )
        body = session.calls[0]["json"]
        self.assertEqual(body["model"], "m")
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["size"], "1024x1024")
        self.assertEqual(body["aspect_ratio"], "1:1")
        self.assertEqual(body["parameters"], {"quality": "high"})
        self.assertEqual(body["references"][0]["mime_type"], "image/png")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://example.invalid/a.png")

    async def test_gemini_inline_data_and_file_uri_parsing(self):
        from imago.core.models import ProviderConfig

        encoded = base64.b64encode(b"png-bytes").decode()
        payload = {
            "candidates": [
                {"content": {"parts": [
                    {"inlineData": {"mimeType": "image/png", "data": encoded}},
                    {"fileData": {"fileUri": "https://example.invalid/file"}},
                ]}},
            ]
        }
        session = _FakeSession(_FakeResponse(200, payload))
        adapter = GeminiAdapter(ProviderConfig("node", "gemini_official", "https://example.invalid", ("key_demo",), model="m"))
        results = await adapter.generate(
            session,
            self._request(references=[ImageInput(b"ref", "image/jpeg", "r.jpg")]),
            "key_demo",
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].data, b"png-bytes")
        self.assertIn("key=key_demo", results[1].url)
        body = session.calls[0]["json"]
        self.assertEqual(body["contents"][0]["parts"][1]["inlineData"]["mimeType"], "image/jpeg")

    async def test_gemini_file_uri_key_already_present(self):
        self.assertEqual(
            _file_uri_with_key("https://example.invalid/f?key=abc", "k"),
            "https://example.invalid/f?key=abc",
        )

    async def test_openai_chat_image_url_success_path(self):
        from imago.core.models import ProviderConfig

        payload = {"choices": [{"message": {"images": [{"image_url": "https://example.invalid/x.png"}]}}]}
        session = _FakeSession(_FakeResponse(200, payload))
        adapter = OpenAIChatAdapter(ProviderConfig("node", "openai_chat", "https://example.invalid", ("key_demo",), model="m"))
        results = await adapter.generate(session, self._request(), "key_demo")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://example.invalid/x.png")

    async def test_openai_chat_empty_image_url_is_skipped(self):
        from imago.core.errors import NoOutputError
        from imago.core.models import ProviderConfig

        payload = {"choices": [{"message": {"images": [{"image_url": ""}]}}]}
        session = _FakeSession(_FakeResponse(200, payload))
        adapter = OpenAIChatAdapter(ProviderConfig("node", "openai_chat", "https://example.invalid", ("key_demo",), model="m"))
        with self.assertRaises(NoOutputError):
            await adapter.generate(session, self._request(), "key_demo")


if __name__ == "__main__":
    unittest.main()
