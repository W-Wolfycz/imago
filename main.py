from __future__ import annotations

import asyncio
from datetime import datetime
import hashlib
import re
import time
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse
from zoneinfo import ZoneInfo

import aiohttp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, Image, Plain, Reply
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, StarTools, register

try:
    from astrbot.core.utils.quoted_message import extract_quoted_message_images
except ImportError:  # AstrBot 旧版本仅处理 Reply.chain 内嵌图片
    extract_quoted_message_images = None

from .core.config import load_config
from .core.errors import QuotaError
from .core.models import DrawTask, GenerationRequest, ImageInput, ImageResult, TaskStage, TaskState
from .core.network import fetch_reference, materialize_result
from .core.prompting import (
    SUMMARY_SYSTEM,
    VISION_SYSTEM,
    compose_persona_prompt,
    optimizer_system,
    persona_optimizer_input,
    summary_user_prompt,
    vision_user_prompt,
)
from .core.security import parse_extra_params, redact, redact_debug
from .integrations.active_send import ProactiveSender, SendOutcome
from .integrations.web_api import PageAPI
from .services.persona_store import PersonaStore
from .services.quota_store import QuotaStore, terminal_refund_amount
from .services.scheduler import TaskScheduler

PENDING_DRAW = "🎨 收到灵感，正在绘制，请稍后…… ✨"
PENDING_PHOTO = "📸 正在为当前人设「{persona}」拍摄，请稍后……"


@register("imago", "Wolfycz", "异步图片生成与 Persona 素材管理", "1.0.1")
class Imago(Star):
    _STAGE_LABELS = {
        TaskStage.QUEUED: "排队中",
        TaskStage.PREPARING_REFERENCES: "正在准备参考图",
        TaskStage.BUILDING_PERSONA: "正在准备 Persona 外观",
        TaskStage.OPTIMIZING_PROMPT: "正在整理本轮画面",
        TaskStage.REQUESTING_PROVIDER: "正在请求图片生成节点",
        TaskStage.PROCESSING_RESULT: "正在处理图片结果",
        TaskStage.DECORATING: "正在执行发送前处理",
        TaskStage.SENDING: "正在发起图片发送",
        TaskStage.FINISHED: "任务流程已结束",
    }
    _STATE_LABELS = {
        TaskState.CREATED: "已创建",
        TaskState.QUEUED: "排队中",
        TaskState.RUNNING: "运行中",
        TaskState.SUCCEEDED: "已生成并完成发送调用",
        TaskState.PARTIAL_SUCCESS: "部分成功并完成发送调用",
        TaskState.DELIVERY_FAILED: "图片已生成，但发送失败",
        TaskState.NO_OUTPUT: "没有可用图片",
        TaskState.FAILED: "已失败",
        TaskState.TIMED_OUT: "已超时",
        TaskState.CANCELLED: "已取消",
    }
    def __init__(self, context: Context, config=None):
        super().__init__(context)
        self.context = context
        self.raw_config = config or {}
        self.session = None
        self.scheduler = None
        self.sender = ProactiveSender(context, self._decorate_error, self._debug)
        data_dir = Path(StarTools.get_data_dir("imago"))
        self.store = PersonaStore(
            data_dir,
            lambda: load_config(self.raw_config).max_upload_bytes,
        )
        self.quota_store = QuotaStore(data_dir, self._quota_today)

    async def initialize(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, connect=15))
        self.scheduler = TaskScheduler(lambda: load_config(self.raw_config), self.session, self._finish_task, self._provider_error, self._scheduler_debug)
        api = PageAPI(self)
        for route, handler, methods, desc in (
            ("/imago/personas", api.personas, ["GET"], "获取 Persona 列表"),
            ("/imago/detail", api.detail, ["GET"], "获取 Persona 素材详情"),
            ("/imago/upload", api.upload, ["POST"], "上传 Persona 参考图"),
            ("/imago/delete", api.delete, ["POST"], "删除 Persona 参考图"),
            ("/imago/preview", api.preview, ["GET"], "预览 Persona 参考图"),
            ("/imago/rebuild", api.rebuild, ["POST"], "重建 Persona 外观摘要"),
            ("/imago/summary", api.set_summary, ["POST"], "手工覆写 Persona 外观摘要"),
            ("/imago/providers", api.providers, ["GET"], "获取图片节点与当前主节点"),
            ("/imago/providers/primary", api.set_primary_provider, ["POST"], "设置主图片生成节点"),
            ("/imago/providers/save", api.save_providers, ["POST"], "保存图片节点配置"),
            ("/imago/quotas", api.quotas, ["GET"], "获取绘图额度列表"),
            ("/imago/quotas/save", api.save_quotas, ["POST"], "保存绘图额度列表"),
        ):
            self.context.register_web_api(route, handler, methods, desc)

    def _quota_today(self) -> str:
        try:
            timezone_name = str(self.context.get_config().get("timezone", "")).strip()
            if timezone_name:
                return datetime.now(ZoneInfo(timezone_name)).date().isoformat()
        except Exception:
            pass
        return datetime.now().astimezone().date().isoformat()

    @staticmethod
    def _event_user_id(event: AstrMessageEvent) -> str:
        return str(event.get_sender_id() or "").strip()

    def _quota_policy(self):
        return load_config(self.raw_config).quota

    def _quota_access(self, event: AstrMessageEvent, amount: int = 1):
        user_id = self._event_user_id(event)
        policy = self._quota_policy()
        if not user_id:
            if policy.enabled:
                raise QuotaError("无法识别用户 ID，不能使用绘图额度。")
            return None
        try:
            return self.quota_store.can_consume(user_id, amount, policy)
        except ValueError as exc:
            raise QuotaError(str(exc)) from exc

    def _remove_unavailable_generation_tools(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        try:
            decision = self._quota_access(event)
        except QuotaError:
            pass
        else:
            if decision is None or decision.allowed:
                return
        if req.func_tool is not None:
            req.func_tool.remove_tool("generate_image")
            req.func_tool.remove_tool("generate_persona_image")

    @staticmethod
    def _quota_target_user_id(event: AstrMessageEvent, target: str) -> str:
        self_id = str(getattr(getattr(event, "message_obj", None), "self_id", "") or "")
        for component in event.get_messages() if hasattr(event, "get_messages") else []:
            if isinstance(component, At):
                value = str(getattr(component, "qq", getattr(component, "target", "")) or "").strip()
                if value and value != self_id:
                    return QuotaStore.normalize_user_id(value)
        return QuotaStore.normalize_user_id(target)

    def _debug(self, message, *args):
        (logger.info if load_config(self.raw_config).debug_to_info else logger.debug)(message, *args)

    def _scheduler_debug(self, task, message, *args):
        safe_args = tuple(
            redact_debug(value) if isinstance(value, (str, dict, list, tuple)) else value
            for value in args
        )
        self._debug(
            self._log_prefix(task.runtime.get("source_event")) + " " + message,
            *safe_args,
        )

    def _log_prefix(self, event=None):
        if load_config(self.raw_config).log_with_bot_id and event is not None:
            try:
                return f"[Imago:{event.get_platform_id()}]"
            except Exception:
                pass
        return "[Imago]"

    def _provider_error(self, provider, exc, task):
        logger.warning("%s 模型失败 type=%s node=%s model=%s status=%s detail=%s", self._log_prefix(task.runtime.get("source_event")), type(exc).__name__, provider.id, provider.model, getattr(exc, "status", None), redact(str(exc)))

    def _decorate_error(self, message, exc):
        logger.error("Imago %s: %s", message, redact(str(exc)))

    @staticmethod
    def _safe_creation_error(exc: Exception) -> str:
        """只向用户和主 LLM 返回任务创建阶段的可公开原因。"""
        message = redact(str(exc)).strip()
        allowed = (
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
        )
        if any(message.startswith(prefix) for prefix in allowed):
            return message
        if isinstance(exc, (TypeError, ValueError, OverflowError)):
            return "任务参数无效"
        return "插件暂时无法创建任务"

    def _tasks_for_event(self, event):
        if not self.scheduler:
            return []
        try:
            return self.scheduler.query(
                umo=str(event.unified_msg_origin or ""),
                owner_user_id=str(event.get_sender_id() or ""),
                bot_instance_id=str(event.get_platform_id() or ""),
            )
        except Exception:
            return []

    def _task_status_lines(self, tasks):
        now = time.monotonic()
        lines = []
        for task in sorted(tasks, key=lambda item: item.created_at):
            kind = "Persona 图片" if task.kind == "persona" else "通用绘图"
            elapsed = max(0, int(now - task.created_at))
            line = (
                f"- 任务 {task.id[:8]} | 类型：{kind} | "
                f"状态：{self._STATE_LABELS.get(task.state, task.state.value)} | "
                f"阶段：{self._STAGE_LABELS.get(task.stage, task.stage.value)} | "
                f"已经过：{elapsed} 秒"
            )
            provider_id = str(task.runtime.get("current_provider_id", ""))
            model = str(task.runtime.get("current_model", ""))
            if provider_id:
                line += f" | 节点：{provider_id}"
            if model:
                line += f" | 模型：{model}"
            lines.append(line)
        return lines

    @filter.on_llm_request(priority=50)
    async def filter_image_generation_tools(self, event: AstrMessageEvent, req: ProviderRequest):
        """不可绘图时仅移除本轮工具，不向 LLM 注入任务或额度信息。"""
        self._remove_unavailable_generation_tools(event, req)

    async def _chat(self, umo: str, system: str, prompt: str, image_urls=None, provider_id_override: str = "", purpose: str = "chat") -> str:
        cfg = load_config(self.raw_config)
        provider_id = provider_id_override or cfg.optimizer_provider_id
        if not provider_id and umo:
            provider_id = await self.context.get_current_chat_provider_id(umo)
        if not provider_id: raise RuntimeError("无可用 Chat Provider")
        self._debug(
            "[Imago] LLM请求 purpose=%s provider=%s images=%d system=%s input=%s",
            purpose, redact_debug(provider_id), len(image_urls or []), redact_debug(system), redact_debug(prompt),
        )
        result = await self.context.llm_generate(chat_provider_id=provider_id, prompt=prompt, system_prompt=system, image_urls=image_urls or [])
        text = getattr(result, "completion_text", None) or getattr(result, "text", None) or str(result)
        text = " ".join(text.split())
        self._debug("[Imago] LLM响应 purpose=%s provider=%s output=%s", purpose, redact_debug(provider_id), redact_debug(text))
        return text

    async def get_persona_prompt(self, persona_id: str) -> str:
        manager = getattr(self.context, "persona_manager", None)
        for method_name in ("get_persona_v3_by_id", "get_persona", "get_persona_by_id"):
            method = getattr(manager, method_name, None)
            if callable(method):
                value = await method(persona_id) if asyncio.iscoroutinefunction(method) else method(persona_id)
                if value: return str(getattr(value, "prompt", "") or (value.get("prompt", "") if isinstance(value, dict) else ""))
        return ""

    async def list_persona_ids(self):
        manager = getattr(self.context, "persona_manager", None)
        values = getattr(manager, "personas_v3", []) or []
        result = set()
        for value in values:
            if isinstance(value, dict):
                persona_id = value.get("persona_id") or value.get("id") or value.get("name")
            else:
                persona_id = getattr(value, "persona_id", getattr(value, "id", ""))
            if persona_id:
                result.add(str(persona_id))
        return sorted(result)

    def _summary_image_urls(self, persona_id: str, reference_names=None):
        names = reference_names or []
        return [str(self.store.reference_path(persona_id, str(name))) for name in names]

    async def _generate_summary(self, persona_id: str, prompt: str, umo: str = "", reference_names=None) -> str:
        images = self._summary_image_urls(persona_id, reference_names)
        vision_provider_id = load_config(self.raw_config).vision_provider_id
        if images and vision_provider_id:
            try:
                evidence = await self._chat(umo, VISION_SYSTEM, vision_user_prompt(len(images)), images, vision_provider_id, "persona_vision")
                summary_input = summary_user_prompt(prompt, evidence)
                summary = await self._chat(umo, SUMMARY_SYSTEM, summary_input, purpose="persona_summary_with_vision")
                return summary
            except Exception as exc:
                self._debug("[Imago] 多模态外观摘要失败，回退纯文本 type=%s", type(exc).__name__)
        summary_input = summary_user_prompt(prompt)
        return await self._chat(umo, SUMMARY_SYSTEM, summary_input, purpose="persona_summary_text")

    async def rebuild_summary(self, persona_id: str, reference_names=None) -> str:
        prompt = await self.get_persona_prompt(persona_id)
        if not prompt: raise ValueError("Persona 不存在或 prompt 为空")
        names = [str(value) for value in (reference_names or [])]
        summary = await self._generate_summary(persona_id, prompt, reference_names=names)
        return self.store.set_summary(persona_id, prompt, summary, manual=False, reference_names=names)["summary"]

    async def _resolve_persona(self, event):
        manager = getattr(self.context, "persona_manager", None)
        resolver = getattr(manager, "resolve_selected_persona", None)
        if not callable(resolver): raise ValueError("当前 AstrBot 不支持 Persona 正式解析")
        umo = event.unified_msg_origin
        conversation_persona_id = None
        try:
            conversation_manager = self.context.conversation_manager
            cid = await conversation_manager.get_curr_conversation_id(umo)
            if cid:
                conversation = await conversation_manager.get_conversation(umo, cid)
                conversation_persona_id = getattr(conversation, "persona_id", None) or None
        except Exception:
            pass
        provider_settings = self.context.get_config(umo=umo).get("provider_settings", {})
        persona_id, persona, _, _ = await resolver(
            umo=umo,
            conversation_persona_id=conversation_persona_id,
            platform_name=event.get_platform_name(),
            provider_settings=provider_settings,
        )
        persona_id = str(persona_id or "")
        prompt = str(getattr(persona, "prompt", "") or (persona.get("prompt", "") if isinstance(persona, dict) else ""))
        if not persona_id or not prompt: raise ValueError("Persona 不存在或 prompt 为空")
        return persona_id, prompt

    async def _submit(self, event, prompt: str, *, persona=False, resolved_persona=None, count=1, aspect_ratio="", size="", extra_params=""):
        if not prompt.strip(): raise ValueError("提示词不能为空")
        count = max(1, min(4, int(count)))
        cfg = load_config(self.raw_config)
        if not cfg.providers: raise ValueError("未配置有效图片节点")
        access = self._quota_access(event, count)
        if access is not None and not access.allowed:
            raise QuotaError(access.reason)
        started_at = time.monotonic()
        task = DrawTask(
            id=uuid.uuid4().hex,
            umo=event.unified_msg_origin,
            request=GenerationRequest(prompt=prompt, count=count, aspect_ratio=aspect_ratio, size=size, extra_params=parse_extra_params(extra_params)),
            owner_user_id=str(event.get_sender_id() or ""),
            bot_instance_id=str(event.get_platform_id() or ""),
            kind="persona" if persona else "draw",
            created_at=started_at,
            updated_at=started_at,
        )
        task.runtime["source_event"] = event
        task.runtime["prepare"] = self._prepare
        task.runtime["finalize"] = self._finalize_task
        task.runtime["primary_provider_id"] = self.store.get_primary_provider_id()
        components = list(event.get_messages()) if hasattr(event, "get_messages") else list(getattr(getattr(event, "message_obj", None), "message", []) or [])
        local_references, deferred_references = self._plan_task_references(components)
        task.request.references.extend(local_references)
        task.runtime["deferred_references"] = deferred_references
        if persona:
            if resolved_persona is None:
                resolved_persona = await self._resolve_persona(event)
            task.persona_id, task.persona_prompt = resolved_persona
        charged = 0
        if task.owner_user_id:
            try:
                decision = self.quota_store.consume(task.owner_user_id, count, cfg.quota)
            except ValueError as exc:
                raise QuotaError(str(exc)) from exc
            if not decision.allowed:
                raise QuotaError(decision.reason)
            charged = decision.charged
        task.runtime["quota_charged"] = charged
        logger.info(
            "%s task=%s 已接管本地参考图 count=%d bytes=%d，待后台解析=%d",
            self._log_prefix(event),
            task.id[:8],
            len(local_references),
            sum(len(item.data) for item in local_references),
            len(deferred_references),
        )
        try:
            return self.scheduler.submit(task)
        except Exception:
            if charged:
                self.quota_store.refund(task.owner_user_id, charged, cfg.quota)
            raise

    @staticmethod
    def _component_local_path(component) -> Path | None:
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

    def _read_local_component(self, component) -> ImageInput | None:
        path = self._component_local_path(component)
        if path is None:
            return None
        data = path.read_bytes()
        if not data or len(data) > load_config(self.raw_config).max_upload_bytes:
            raise ValueError("图片格式或大小不符合要求")
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(path.suffix.lower())
        if not mime:
            raise ValueError("图片格式或大小不符合要求")
        return ImageInput(data=data, mime_type=mime, filename=path.name)

    def _plan_task_references(self, components):
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
            values = [getattr(component, "message_str", "")]
            for item in getattr(component, "chain", None) or []:
                values.append(getattr(item, "text", "") or getattr(item, "content", ""))
            text = " ".join(str(value) for value in values if value)
            return any(marker in text for marker in ("[图片]", "[Image]", "[image]"))

        def walk(chain) -> None:
            for component in chain or []:
                name = type(component).__name__.lower()
                if name == "reply":
                    reply_chain = getattr(component, "chain", None) or []
                    has_embedded_image = contains_image(reply_chain)
                    walk(reply_chain)
                    if not has_embedded_image and extract_quoted_message_images is not None:
                        deferred.append({"kind": "reply", "component": component, "strict": reply_indicates_image(component)})
                    continue
                if "image" in name:
                    image = self._read_local_component(component)
                    if image is not None:
                        add_local(image)
                        continue
                    source = next((getattr(component, field, None) for field in ("url", "file", "path") if getattr(component, field, None)), None)
                    deferred.append({"kind": "source", "source": str(source or ""), "component": component, "strict": True})
                    continue
                text = getattr(component, "text", None) or getattr(component, "content", None)
                if isinstance(text, str):
                    for url in re.findall(r"https?://[^\s<>\]\[()\"']+", text):
                        deferred.append({"kind": "source", "source": url.rstrip(".,;!?。，；！？"), "strict": False})

        walk(components)
        return local_references, deferred

    async def _resolve_deferred_references(self, task: DrawTask) -> None:
        pending = list(task.runtime.pop("deferred_references", []) or [])
        event = task.runtime.get("source_event")
        seen = {hashlib.sha256(item.data).digest() for item in task.request.references}

        async def add_source(source, *, strict=True):
            try:
                image = await fetch_reference(
                    self.session,
                    str(source),
                    max_bytes=load_config(self.raw_config).max_upload_bytes,
                    block_private=load_config(self.raw_config).block_private_networks,
                )
                digest = hashlib.sha256(image.data).digest()
                if digest not in seen:
                    seen.add(digest)
                    task.request.references.append(image)
            except Exception as exc:
                self._debug("%s task=%s 后台参考图解析失败 type=%s", self._log_prefix(event), task.id[:8], type(exc).__name__)
                if strict:
                    raise

        for item in pending:
            if item.get("kind") == "reply":
                if event is None or extract_quoted_message_images is None:
                    continue
                sources = await extract_quoted_message_images(event, item.get("component"))
                if not sources and item.get("strict"):
                    raise ValueError("引用消息图片无法获取")
                for source in sources:
                    await add_source(source, strict=True)
                continue
            source = str(item.get("source", ""))
            if not source:
                component = item.get("component")
                converter = getattr(component, "convert_to_file_path", None)
                if not callable(converter):
                    raise ValueError("图片组件没有可读取来源")
                source = await converter()
            await add_source(source, strict=bool(item.get("strict", True)))

    async def _event_references(self, components, event=None):
        references = []
        seen = set()

        def contains_image(chain):
            for item in chain or []:
                item_name = type(item).__name__.lower()
                if "image" in item_name:
                    return True
                if item_name == "reply" and contains_image(getattr(item, "chain", None) or []):
                    return True
            return False

        async def add_source(source, *, strict=False):
            try:
                image = await fetch_reference(self.session, str(source), max_bytes=load_config(self.raw_config).max_upload_bytes, block_private=load_config(self.raw_config).block_private_networks)
                digest = hashlib.sha256(image.data).digest()
                if digest not in seen:
                    seen.add(digest)
                    references.append(image)
                return True
            except Exception as exc:
                self._debug("%s 忽略无效消息参考图 type=%s", self._log_prefix(event), type(exc).__name__)
                if strict:
                    raise
                return False

        async def walk(chain):
            for component in chain or []:
                name = type(component).__name__.lower()
                if name == "reply":
                    reply_chain = getattr(component, "chain", None) or []
                    has_embedded_image = contains_image(reply_chain)
                    await walk(reply_chain)
                    if not has_embedded_image and event is not None and extract_quoted_message_images is not None:
                        quoted_sources = await extract_quoted_message_images(event, component)
                        for source in quoted_sources:
                            await add_source(source, strict=True)
                    continue
                if "image" in name:
                    converter = getattr(component, "convert_to_file_path", None)
                    if callable(converter):
                        try:
                            local_path = await converter()
                            await add_source(local_path, strict=True)
                            continue
                        except Exception as exc:
                            self._debug("%s 图片本地化失败，尝试原始来源 type=%s", self._log_prefix(event), type(exc).__name__)
                    source = next((getattr(component, field, None) for field in ("url", "file", "path") if getattr(component, field, None)), None)
                    if source:
                        await add_source(source, strict=True)
                    else:
                        raise ValueError("图片组件没有可读取来源")
                    continue
                text = getattr(component, "text", None) or getattr(component, "content", None)
                if isinstance(text, str):
                    for url in re.findall(r"https?://[^\s<>\]\[()\"']+", text):
                        await add_source(url.rstrip(".,;!?。，；！？"))

        await walk(components)
        return references

    async def _prepare(self, task: DrawTask):
        cfg = load_config(self.raw_config)
        self.store.update_task_manifest(task.id, kind=task.kind, state="running", stage=TaskStage.PREPARING_REFERENCES.value)
        await self._resolve_deferred_references(task)
        self.store.persist_task_inputs(task.id, task.request.references)
        task.runtime["persona_references"] = []
        if task.persona_id:
            self.scheduler.set_stage(task, TaskStage.BUILDING_PERSONA)
            cached = self.store.get_summary(task.persona_id, task.persona_prompt)
            summary = cached["summary"] if cached else (await self._generate_summary(task.persona_id, task.persona_prompt, task.umo))
            if not cached: self.store.set_summary(task.persona_id, task.persona_prompt, summary, manual=False)
            dynamic = task.request.prompt
            if cfg.optimizer_enabled:
                self.scheduler.set_stage(task, TaskStage.OPTIMIZING_PROMPT)
                optimizer_system_prompt = optimizer_system(cfg.optimizer_prompt, cfg.optimizer_style, persona=True)
                optimizer_input = persona_optimizer_input(summary, dynamic)
                dynamic = await self._chat(
                    task.umo,
                    optimizer_system_prompt,
                    optimizer_input,
                    purpose="persona_scene_optimizer",
                )
            task.request.prompt = compose_persona_prompt(summary, dynamic)
            persona_references = []
            for ref in self.store.list_references(task.persona_id):
                path = self.store.reference_path(task.persona_id, str(ref["name"]))
                mime = {".png":"image/png",".jpg":"image/jpeg",".webp":"image/webp",".gif":"image/gif"}.get(path.suffix.lower(), "image/png")
                persona_references.append(ImageInput(path.read_bytes(), mime, path.name))
            task.runtime["persona_references"] = persona_references
        self._debug(
            "%s task=%s 最终生成请求 kind=%s prompt=%s params=%s",
            self._log_prefix(task.runtime.get("source_event")),
            task.id[:8],
            task.kind,
            redact_debug(task.request.prompt),
            redact_debug(task.request.extra_params),
        )
        logger.info(
            "%s task=%s 参考图准备完成 explicit_count=%d persona_pool=%d bytes=%d",
            self._log_prefix(task.runtime.get("source_event")),
            task.id[:8],
            len(task.request.references),
            len(task.runtime.get("persona_references", [])),
            sum(len(item.data) for item in [*task.request.references, *task.runtime.get("persona_references", [])]),
        )

    async def _finish_task(self, task: DrawTask, results: list[ImageResult]) -> SendOutcome:
        if task.state == TaskState.CANCELLED or not self.scheduler.accepting:
            return SendOutcome(
                False,
                "Cancelled",
                generation_success=False,
                usable_output_count=0,
            )
        self.scheduler.set_stage(task, TaskStage.PROCESSING_RESULT)
        chain = []
        output_paths = []
        task.runtime["provider_output_count"] = len(results)
        output_dir = self.store.task_output_dir(task.id)
        for result in results:
            try:
                value = await materialize_result(self.session, result, output_dir, max_bytes=load_config(self.raw_config).max_upload_bytes, block_private=load_config(self.raw_config).block_private_networks)
                if isinstance(value, Path): output_paths.append(value)
                chain.append(Image.fromFileSystem(str(value)) if isinstance(value, Path) else Image.fromURL(value))
                self._debug("%s task=%s 生成结果已落盘 index=%d bytes=%d", self._log_prefix(task.runtime.get("source_event")), task.id[:8], len(chain), value.stat().st_size if isinstance(value, Path) else 0)
            except Exception as exc: task.errors.append(type(exc).__name__)
        if output_paths:
            self.store.record_task_outputs(task.id, output_paths)
        usable_output_count = len(chain)
        generation_success = usable_output_count > 0
        task.runtime["generation_success"] = generation_success
        task.runtime["usable_output_count"] = usable_output_count
        task.runtime["output_processing_completed"] = True
        event = task.runtime.get("source_event")
        delivery_kind = "image" if generation_success else "notification"
        delivery_prefix = f"{delivery_kind}_delivery"
        if event is None:
            task.runtime[f"{delivery_prefix}_attempted"] = False
            task.runtime[f"{delivery_prefix}_success"] = False
            task.runtime[f"{delivery_prefix}_error"] = "MissingSourceEvent"
            return SendOutcome(
                False,
                "MissingSourceEvent",
                generation_success=generation_success,
                usable_output_count=usable_output_count,
                delivery_kind=delivery_kind,
            )
        if not generation_success:
            message_id = getattr(getattr(event, "message_obj", None), "message_id", None)
            if message_id:
                chain.append(Reply(id=message_id, sender_id=event.get_sender_id()))
            chain.append(Plain("绘制超时，请稍后再试。" if task.state == TaskState.TIMED_OUT else "绘制失败，请稍后再试。"))
        elif task.errors:
            chain.append(Plain("部分图片绘制失败。"))
        self.scheduler.set_stage(task, TaskStage.DECORATING)
        task.runtime[f"{delivery_prefix}_attempted"] = True
        send_outcome = await self.sender.send(
            task.umo,
            event,
            chain,
            before_send=lambda: self.scheduler.set_stage(task, TaskStage.SENDING),
        )
        task.runtime[f"{delivery_prefix}_success"] = send_outcome.success
        task.runtime[f"{delivery_prefix}_error"] = send_outcome.error
        task.runtime[f"{delivery_prefix}_side_effects_started"] = send_outcome.side_effects_started
        task.runtime[f"{delivery_prefix}_side_send_started"] = send_outcome.side_send_started
        task.runtime[f"{delivery_prefix}_side_send_error"] = send_outcome.side_send_error
        if not send_outcome.success:
            logger.error(
                "%s task=%s generation_success=%s delivery=%s 主动发送失败 type=%s",
                self._log_prefix(event),
                task.id[:8],
                generation_success,
                delivery_kind,
                send_outcome.error or "Unknown",
            )
        return SendOutcome(
            send_outcome.success,
            send_outcome.error,
            side_effects_started=send_outcome.side_effects_started,
            side_send_started=send_outcome.side_send_started,
            side_send_error=send_outcome.side_send_error,
            generation_success=generation_success,
            usable_output_count=usable_output_count,
            delivery_kind=delivery_kind,
        )

    def _settle_task_quota(self, task: DrawTask) -> None:
        charged = max(0, int(task.runtime.get("quota_charged", 0) or 0))
        refund_amount = terminal_refund_amount(task.state, charged)
        task.runtime["quota_refund_eligible"] = refund_amount > 0
        task.runtime.setdefault("quota_refund_attempted", False)
        task.runtime.setdefault("quota_refund_success", False)
        task.runtime.setdefault("quota_refunded_amount", 0)
        task.runtime.setdefault("quota_refund_error", "")
        if refund_amount <= 0 or task.runtime.get("quota_refund_success"):
            return

        task.runtime["quota_refund_attempted"] = True
        try:
            self.quota_store.refund(
                task.owner_user_id,
                refund_amount,
                self._quota_policy(),
            )
        except Exception as exc:
            error_type = type(exc).__name__
            task.runtime["quota_refund_error"] = error_type
            marker = f"quota_refund:{error_type}"
            if marker not in task.errors:
                task.errors.append(marker)
            logger.error(
                "%s task=%s 绘图额度退回失败 amount=%d type=%s",
                self._log_prefix(task.runtime.get("source_event")),
                task.id[:8],
                refund_amount,
                error_type,
            )
            return

        task.runtime["quota_refund_success"] = True
        task.runtime["quota_refunded_amount"] = refund_amount
        task.runtime["quota_refund_error"] = ""
        logger.info(
            "%s task=%s 绘图额度已退回 amount=%d state=%s",
            self._log_prefix(task.runtime.get("source_event")),
            task.id[:8],
            refund_amount,
            task.state.value,
        )

    async def _finalize_task(self, task: DrawTask) -> None:
        def delivery(prefix: str) -> dict:
            attempted = bool(task.runtime.get(f"{prefix}_attempted", False))
            return {
                "attempted": attempted,
                "success": bool(task.runtime.get(f"{prefix}_success", False)) if attempted else None,
                "error": str(task.runtime.get(f"{prefix}_error", "") or ""),
                "side_effects_started": bool(task.runtime.get(f"{prefix}_side_effects_started", False)),
                "side_send_started": bool(task.runtime.get(f"{prefix}_side_send_started", False)),
                "side_send_error": str(task.runtime.get(f"{prefix}_side_send_error", "") or ""),
            }

        self._settle_task_quota(task)
        charged = max(0, int(task.runtime.get("quota_charged", 0) or 0))
        refund_attempted = bool(task.runtime.get("quota_refund_attempted", False))
        self.store.update_task_manifest(
            task.id,
            kind=task.kind,
            state=task.state.value,
            stage=task.stage.value,
            requested_output_count=task.request.count,
            provider_output_count=max(0, int(task.runtime.get("provider_output_count", 0) or 0)),
            usable_output_count=max(0, int(task.runtime.get("usable_output_count", 0) or 0)),
            generation_success=bool(task.runtime.get("generation_success", False)),
            image_delivery=delivery("image_delivery"),
            notification_delivery=delivery("notification_delivery"),
            quota={
                "charged": charged,
                "refund_eligible": bool(task.runtime.get("quota_refund_eligible", False)),
                "refund_attempted": refund_attempted,
                "refund_success": bool(task.runtime.get("quota_refund_success", False)) if refund_attempted else None,
                "refunded_amount": max(0, int(task.runtime.get("quota_refunded_amount", 0) or 0)),
                "refund_error": str(task.runtime.get("quota_refund_error", "") or ""),
            },
            errors=list(task.errors),
        )
        if task.state == TaskState.CANCELLED:
            return
        active = set(self.scheduler.tasks) if self.scheduler else set()
        self.store.prune_task_cache(load_config(self.raw_config).temp_cache_bytes, protected=active)

    @filter.command_group("imago")
    def imago_group(self): pass

    @imago_group.group("quota")
    def quota_group(self): pass

    @quota_group.command("help")
    async def quota_help(self, event: AstrMessageEvent):
        yield event.plain_result(
            "/imago quota show：查看自己的额度\n"
            "/imago quota sign：每日签到领取额度\n"
            "/imago quota add/del/set <用户 ID> <整数>：管理员调整额度"
        )

    @quota_group.command("show")
    async def quota_show(self, event: AstrMessageEvent):
        try:
            snapshot = self.quota_store.inspect(self._event_user_id(event), self._quota_policy())
            if snapshot.blocked:
                message = "当前无法使用绘图功能。"
            elif snapshot.unlimited:
                message = "当前为无限额度。"
            elif not snapshot.quota_enabled:
                message = f"绘图额度功能未启用；已记录余额为 {snapshot.quota}。"
            else:
                message = f"当前剩余绘图额度：{snapshot.quota}。"
            if snapshot.last_checkin_date:
                message += f" 最近签到日期：{snapshot.last_checkin_date}。"
            yield event.plain_result(message)
        except Exception as exc:
            yield event.plain_result(self._safe_creation_error(exc))

    @quota_group.command("sign")
    async def quota_sign(self, event: AstrMessageEvent):
        try:
            result = self.quota_store.checkin(self._event_user_id(event), self._quota_policy())
            if not result.success:
                yield event.plain_result(result.reason)
                return
            yield event.plain_result(
                f"签到成功，获得 {result.reward} 点绘图额度；当前余额 {result.snapshot.quota}。"
            )
        except Exception as exc:
            yield event.plain_result(self._safe_creation_error(exc))

    async def _quota_admin_adjust(self, event: AstrMessageEvent, operation: str, target: str, amount):
        try:
            if not str(target or "").strip() or amount is None:
                raise ValueError("请提供用户 ID 和额度整数")
            try:
                amount = int(amount)
            except (TypeError, ValueError) as exc:
                raise ValueError("额度必须是整数") from exc
            user_id = self._quota_target_user_id(event, target)
            snapshot = self.quota_store.adjust(user_id, operation, amount, self._quota_policy())
            action = {"add": "增加", "del": "扣除", "set": "设置"}[operation]
            return f"已为用户 {user_id} {action}额度 {max(0, int(amount))}；当前余额 {snapshot.quota}。"
        except Exception as exc:
            return self._safe_creation_error(exc)

    @filter.permission_type(filter.PermissionType.ADMIN)
    @quota_group.command("add")
    async def quota_add(self, event: AstrMessageEvent, target: str = "", amount=None):
        yield event.plain_result(await self._quota_admin_adjust(event, "add", target, amount))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @quota_group.command("del")
    async def quota_del(self, event: AstrMessageEvent, target: str = "", amount=None):
        yield event.plain_result(await self._quota_admin_adjust(event, "del", target, amount))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @quota_group.command("set")
    async def quota_set(self, event: AstrMessageEvent, target: str = "", amount=None):
        yield event.plain_result(await self._quota_admin_adjust(event, "set", target, amount))

    @imago_group.command("help")
    async def help_command(self, event: AstrMessageEvent):
        lines = [
            "🖼️ IMAGO·映相 指令帮助",
            "",
            "[开始绘图]",
            "- /imago draw <画面描述>  普通绘图或参考图改图",
            "- /画 <画面描述>  普通绘图快捷入口",
            "- /imago photo <画面要求>  让当前人设出镜",
            "- /拍照 <画面要求>  人设出镜快捷入口",
            "",
            "[任务与额度]",
            "- /imago status  查看正在处理或刚结束的任务",
            "- /imago quota show  查看自己的绘图额度",
            "- /imago quota sign  每日签到领取额度",
            "",
            "[Persona 素材]",
            "- /imago ref-upload  为当前人设上传参考图（同条消息附图）",
            "- /imago summary-show  查看当前人设外观摘要",
            "- /imago summary-rebuild  重新生成外观摘要",
            "- /imago summary-set <外观摘要>  手工设置外观摘要",
            "",
            "[图片节点]",
            "- /imago provider-primary <节点 ID>  设置主节点（也可在 WebUI 设置）",
        ]
        if event.is_admin():
            lines.extend([
                "",
                "[Bot 管理员]",
                "- /imago quota add <用户 ID> <整数>  增加额度",
                "- /imago quota del <用户 ID> <整数>  扣除额度",
                "- /imago quota set <用户 ID> <整数>  设置额度",
            ])
        lines.extend([
            "",
            "[示例]",
            "- /画 雨夜街头的橘猫，电影感，横幅构图",
            "- /拍照 在海边回头看向镜头，黄昏逆光",
        ])

        md_text = "\n".join(lines)
        try:
            image_path = await self.text_to_image(md_text, return_url=False)
            yield event.image_result(image_path)
        except Exception as exc:
            logger.warning(
                "%s 帮助菜单 T2I 失败，回退纯文本 type=%s",
                self._log_prefix(event),
                type(exc).__name__,
            )
            yield event.plain_result(md_text)

    @imago_group.command("status")
    async def status_command(self, event: AstrMessageEvent):
        tasks = self._tasks_for_event(event)
        if not tasks:
            yield event.plain_result("当前没有正在处理或刚刚结束的映相任务。")
            return
        yield event.plain_result("当前映相任务：\n" + "\n".join(self._task_status_lines(tasks)) + "\n不提供准确完成时间。")

    @imago_group.command("draw")
    async def draw_command(self, event: AstrMessageEvent, prompt: str):
        try: await self._submit(event, prompt); yield event.plain_result(PENDING_DRAW)
        except Exception as exc: yield event.plain_result(f"无法创建任务：{self._safe_creation_error(exc)}")

    @imago_group.command("photo")
    async def photo_command(self, event: AstrMessageEvent, action: str):
        try:
            resolved_persona = await self._resolve_persona(event)
            await self._submit(event, action, persona=True, resolved_persona=resolved_persona)
            yield event.plain_result(PENDING_PHOTO.format(persona=resolved_persona[0]))
        except Exception as exc: yield event.plain_result(f"无法创建任务：{self._safe_creation_error(exc)}")

    @filter.command("画")
    async def draw_shortcut(self, event: AstrMessageEvent, prompt: str):
        async for result in self.draw_command(event, prompt): yield result

    @filter.command("拍照")
    async def photo_shortcut(self, event: AstrMessageEvent, action: str):
        async for result in self.photo_command(event, action): yield result

    @imago_group.command("summary-show")
    async def summary_show(self, event: AstrMessageEvent):
        try:
            persona_id, prompt = await self._resolve_persona(event); item = self.store.get_summary(persona_id, prompt)
            yield event.plain_result((item or {}).get("summary", "尚无可用外观摘要。"))
        except Exception as exc: yield event.plain_result(redact(str(exc)))

    @imago_group.command("summary-rebuild")
    async def summary_rebuild(self, event: AstrMessageEvent):
        persona_id, _ = await self._resolve_persona(event); yield event.plain_result(await self.rebuild_summary(persona_id))

    @imago_group.command("summary-set")
    async def summary_set(self, event: AstrMessageEvent, summary: str):
        persona_id, prompt = await self._resolve_persona(event); self.store.set_summary(persona_id, prompt, summary, manual=True); yield event.plain_result("已设置手工外观摘要。")

    @imago_group.command("ref-upload")
    async def ref_upload(self, event: AstrMessageEvent):
        try:
            persona_id, _ = await self._resolve_persona(event)
            components = list(event.get_messages()) if hasattr(event, "get_messages") else []
            images = await self._event_references(components, event)
            if not images: raise ValueError("请在同一条消息中附带图片")
            for image in images: self.store.add_reference(persona_id, image.data, image.mime_type)
            yield event.plain_result(f"已上传 {len(images)} 张 Persona 参考图。")
        except Exception as exc: yield event.plain_result(redact(str(exc)))

    @imago_group.command("provider-primary")
    async def provider_primary(self, event: AstrMessageEvent, provider_id: str):
        configured = [item.id for item in load_config(self.raw_config).providers]
        provider_id = provider_id.strip()
        if provider_id not in configured:
            yield event.plain_result(f"节点不存在。已配置节点：{', '.join(configured) or '无'}")
            return
        self.store.set_primary_provider_id(provider_id)
        yield event.plain_result(f"已将主图片生成节点设为 {provider_id}；其他节点将按配置顺序 fallback。")

    @filter.llm_tool(name="generate_image")
    async def generate_image(self, event: AstrMessageEvent, prompt: str, count: int = 1, aspect_ratio: str = "", size: str = "", extra_params: str = ""):
        """
        当用户要求新生成、绘制、改图或重绘一张图片，且当前会话 Persona 本人不需要出现在画面中时，必须调用本工具。
        不要只用文字描述成图、假装已经画好，或在未成功创建任务时声称稍后会发图。

        适用于场景、物品、海报、非当前会话 Persona 角色或其他普通图片。
        当前会话 Persona 本人需要出镜时应改用 generate_persona_image。
        用户只是讨论、评价或询问已有图片时不要调用。
        用户当前消息、引用消息及消息正文中可访问的 HTTP/HTTPS 图片会自动作为本轮参考图；
        插件会等待这些图片完成本地化后再请求图片节点。工具只创建后台任务，不等待生成完成。
        不得承诺完成百分比、排队名次或准确完成时间。

        Args:
            prompt(string): 完整、可直接交给图片模型的画面提示词。插件不再调用副脑改写。
            count(int): 生成数量，范围 1-4。
            aspect_ratio(string): 可选宽高比，如 1:1、16:9。
            size(string): 可选尺寸，如 1024x1024；与宽高比冲突时以 size 为准。
            extra_params(string): 只能填写用户明确提供的 --key value 参数；用户没有指定时留空。
        """
        try:
            await self._submit(event, prompt, count=count, aspect_ratio=aspect_ratio, size=size, extra_params=extra_params)
            return "后台绘图任务已创建，但图片目前尚未生成或确认送达；插件会在处理完成后另行发送。不要承诺准确完成时间。请以当前 Persona 的语气简短回复用户，自然表达“收到灵感，正在绘制，请稍等一下”。"
        except Exception as exc:
            return f"后台绘图任务未能创建。可告知用户的原因：{self._safe_creation_error(exc)}。请以当前 Persona 的语气简短说明失败，不要虚构任务已开始或图片已生成。"

    @filter.llm_tool(name="generate_persona_image")
    async def generate_persona_image(self, event: AstrMessageEvent, action: str, count: int = 1, aspect_ratio: str = "", size: str = "", extra_params: str = ""):
        """
        当用户要求当前会话 Persona 自拍、拍照、发一张本人照片、以图片展示动作或场景、合影，或以其他方式本人出镜时，必须调用本工具。
        不要只用文字扮演拍照、假装已经拍好，或在未成功创建任务时声称稍后会发照片。

        适用于自拍、他拍、第三人称场景照、全身照、特写、合影或其他需要当前 Persona 出镜的画面。
        当前会话 Persona 本人不需要出镜的普通绘图应改用 generate_image。
        用户只是讨论、评价或询问已有照片时不要调用。
        不要在 action 中重复 Persona 的稳定外貌；插件会自动加入外观摘要和 Persona 固定参考图。
        用户当前消息、引用消息及消息正文中可访问的 HTTP/HTTPS 图片也会自动加入本轮参考图；
        插件会等待这些图片完成本地化后再请求图片节点。工具只创建后台任务，不等待生成完成。
        不得承诺完成百分比、排队名次或准确完成时间。

        Args:
            action(string): 本轮动态画面需求：动作、场景、临时服饰、视角、构图、镜头和光线。
            count(int): 生成数量，范围 1-4。
            aspect_ratio(string): 可选宽高比，如 1:1、16:9。
            size(string): 可选尺寸，如 1024x1024；与宽高比冲突时以 size 为准。
            extra_params(string): 只能填写用户明确提供的 --key value 参数；用户没有指定时留空。
        """
        try:
            await self._submit(event, action, persona=True, count=count, aspect_ratio=aspect_ratio, size=size, extra_params=extra_params)
            return "后台 Persona 图片任务已创建，但图片目前尚未生成或确认送达；插件会在处理完成后另行发送。不要承诺准确完成时间。请以当前 Persona 的语气简短回复用户，自然表达“正在拍摄，请稍后……”；不要声称已经拍好。"
        except Exception as exc:
            return f"后台 Persona 图片任务未能创建。可告知用户的原因：{self._safe_creation_error(exc)}。请以当前 Persona 的语气简短说明失败，不要虚构任务已开始、正在拍摄或图片已经生成。"

    async def terminate(self):
        if self.scheduler: await self.scheduler.close()
        if self.session: await self.session.close()
