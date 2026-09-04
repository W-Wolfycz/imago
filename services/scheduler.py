from __future__ import annotations

import asyncio
import random
import time
import uuid
from contextlib import asynccontextmanager
from collections.abc import Awaitable, Callable
from dataclasses import replace

try:
    from aiohttp import ClientError as _AiohttpClientError
except ImportError:  # 允许在不含 AstrBot 运行依赖的源码环境运行纯逻辑测试
    class _AiohttpClientError(Exception):
        pass

from ..core.errors import ConfigurationError, NoOutputError, ProviderError
from ..core.models import DrawTask, ImageResult, TaskStage, TaskState
from ..core.prompting import REFERENCE_RELATION_SUFFIX, reference_relation_suffix
from ..core.security import redact_debug
from ..providers import ADAPTERS


class _DynamicLimiter:
    """动态并发闸门；降低上限时不取消已运行任务。"""

    def __init__(self, limit_getter):
        self._limit_getter = limit_getter
        self._active = 0
        self._condition = asyncio.Condition()

    def _limit(self) -> int:
        return max(1, int(self._limit_getter()))

    @property
    def active(self) -> int:
        return self._active

    async def acquire(self) -> None:
        async with self._condition:
            while self._active >= self._limit():
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=0.1)
                except asyncio.TimeoutError:
                    # 配置对象可能由 AstrBot 管理页原地更新；轮询让“提高上限”
                    # 无需等待已有任务释放槽位即可生效。
                    pass
            self._active += 1

    async def release(self) -> None:
        async with self._condition:
            self._active = max(0, self._active - 1)
            self._condition.notify_all()

    @asynccontextmanager
    async def slot(self):
        await self.acquire()
        try:
            yield
        finally:
            await self.release()


class TaskScheduler:
    def __init__(self, config_getter, session, runner: Callable[[DrawTask, list[ImageResult]], Awaitable[bool]], logger, debug=None):
        self.config_getter = config_getter
        self.session = session
        self.runner = runner
        self.logger = logger
        self.debug = debug or (lambda *args, **kwargs: None)
        self.accepting = True
        self.tasks: dict[str, tuple[DrawTask, asyncio.Task]] = {}
        self.recent: dict[str, tuple[DrawTask, float]] = {}
        self._limiter = _DynamicLimiter(
            lambda: self.config_getter().max_concurrent_tasks,
        )
        self._key_indices: dict[str, int] = {}

    def submit(self, task: DrawTask) -> str:
        if not self.accepting: raise RuntimeError("插件正在关闭")
        task.id = task.id or uuid.uuid4().hex
        now = time.monotonic()
        task.created_at = task.created_at or now
        task.updated_at = now
        task.state = TaskState.QUEUED
        task.stage = TaskStage.QUEUED
        handle = asyncio.create_task(self._execute(task), name=f"imago:{task.id}")
        self.tasks[task.id] = (task, handle)
        handle.add_done_callback(lambda _: self._archive(task.id))
        return task.id

    def _archive(self, task_id: str) -> None:
        item = self.tasks.pop(task_id, None)
        if not item:
            return
        task, _ = item
        if task.state != TaskState.CANCELLED:
            self.recent[task_id] = (task, time.monotonic() + 30.0)
        self._prune_recent()

    def _prune_recent(self) -> None:
        now = time.monotonic()
        for task_id, (_, expires_at) in list(self.recent.items()):
            if expires_at <= now:
                self.recent.pop(task_id, None)

    @staticmethod
    def _request_for_attempt(task: DrawTask, provider):
        """Build the request for one model attempt, sampling Persona references if configured."""
        explicit_references = list(task.request.references)
        persona_references = list(task.runtime.get("persona_references", []) or [])
        limit = max(0, int(getattr(provider, "reference_image_limit", 0) or 0))
        if limit > 0:
            persona_budget = max(0, limit - len(explicit_references))
            selected_persona = random.sample(
                persona_references,
                min(persona_budget, len(persona_references)),
            ) if persona_budget else []
        else:
            selected_persona = persona_references
        references = [*explicit_references, *selected_persona]
        prompt = task.request.prompt
        if explicit_references and REFERENCE_RELATION_SUFFIX not in prompt:
            # 低优先级关系声明按本节点实际采样数量追加：前 N 张是用户指定参考、
            # 其余是人设固定图，避免模型在图片较多时默认跟随人设图（按全量池
            # 计数会在 reference_image_limit 采样后数量虚高）。
            prompt = (
                f"{prompt}\n\n"
                f"{reference_relation_suffix(len(explicit_references), len(selected_persona))}"
            )
        task.runtime["current_explicit_reference_count"] = len(explicit_references)
        task.runtime["current_persona_reference_count"] = len(selected_persona)
        task.runtime["current_reference_count"] = len(references)
        return replace(task.request, references=references, prompt=prompt)

    @staticmethod
    def set_stage(task: DrawTask, stage: TaskStage) -> None:
        task.stage = stage
        task.updated_at = time.monotonic()

    def query(self, *, umo: str, owner_user_id: str, bot_instance_id: str) -> list[DrawTask]:
        self._prune_recent()
        values = [task for task, _ in self.tasks.values()]
        values.extend(task for task, _ in self.recent.values())
        return [
            task for task in values
            if task.umo == umo
            and task.owner_user_id == owner_user_id
            and task.bot_instance_id == bot_instance_id
        ]

    async def _execute(self, task: DrawTask) -> None:
        config = self.config_getter()
        try:
            remaining = config.generation_timeout - max(0.0, time.monotonic() - task.created_at)
            if remaining <= 0:
                raise asyncio.TimeoutError
            self.debug(
                task,
                "task=%s 开始执行 kind=%s timeout=%.0fs queued=%.1fs request_count=%d",
                task.id[:8], task.kind, remaining,
                max(0.0, time.monotonic() - task.created_at),
                task.request.count,
            )
            async with asyncio.timeout(remaining):
                async with self._limiter.slot():
                    task.state = TaskState.RUNNING
                    self.set_stage(task, TaskStage.PREPARING_REFERENCES)
                    prepare = task.runtime.get("prepare")
                    if prepare:
                        await prepare(task)
                    self.set_stage(task, TaskStage.REQUESTING_PROVIDER)
                    results = await self._generate(task)
                    task.runtime["provider_output_count"] = len(results)
                    if len(results) < task.request.count:
                        task.errors.append(f"missing_results:{task.request.count - len(results)}")
                    task.runtime["runner_started"] = True
                    outcome = await self.runner(task, results)
                    generation_success = self._record_outcome(task, outcome, results)
                    sent = bool(getattr(outcome, "success", outcome))
                    if not generation_success:
                        task.state = TaskState.NO_OUTPUT
                    elif sent:
                        task.state = TaskState.SUCCEEDED if not task.errors else TaskState.PARTIAL_SUCCESS
                    else:
                        task.state = TaskState.DELIVERY_FAILED
                        error = str(getattr(outcome, "error", "") or "")
                        if error:
                            task.errors.append(f"delivery:{error}")
        except asyncio.TimeoutError:
            task.state = TaskState.TIMED_OUT
            task.runtime.setdefault("generation_success", False)
            task.runtime.setdefault("usable_output_count", 0)
            # 供失败配文透传原因（脱敏后转述给用户）。
            task.runtime.setdefault("last_provider_error", "任务超时")
            # 超时时若主发送流程尚未完成（runner 未运行，或已运行但平台发送没
            # 走完且没有成功投递记录），补发一次超时通知：配文/发送把预算耗尽时
            # 用户不应静默收不到任何消息。runner 已完成主发送（_finish_task 已置
            # runner_send_completed）则不再补发，避免平台已收到主消息又被终态
            # 通知重复打扰。_send_terminal_notice 会以空结果重跑 _finish_task
            # （预算已尽的失败通知路径，配文会因 remaining 不足直接跳过，不会
            # 再次拖超时）。
            delivery_success = bool(
                task.runtime.get("image_delivery_success", False)
                or task.runtime.get("notification_delivery_success", False)
            )
            send_completed = bool(task.runtime.get("runner_send_completed", False))
            if not send_completed and (
                not task.runtime.get("runner_started") or not delivery_success
            ):
                await self._send_terminal_notice(task)
        except asyncio.CancelledError:
            task.state = TaskState.CANCELLED
            raise
        except Exception as exc:
            task.errors.append(type(exc).__name__)
            # 供失败配文透传原因：脱敏 + 截断，避免上游任意文本直接进 LLM 上下文。
            task.runtime["last_provider_error"] = redact_debug(str(exc))[:200] or type(exc).__name__
            if task.runtime.get("generation_success"):
                task.state = TaskState.DELIVERY_FAILED
            elif isinstance(exc, NoOutputError):
                task.state = TaskState.NO_OUTPUT
            elif task.runtime.get("output_processing_completed"):
                task.state = TaskState.NO_OUTPUT
            else:
                task.state = TaskState.FAILED
            task.runtime.setdefault("generation_success", False)
            task.runtime.setdefault("usable_output_count", 0)
            if not task.runtime.get("runner_started"):
                await self._send_terminal_notice(task)
        finally:
            self.set_stage(task, TaskStage.FINISHED)
            finalize = task.runtime.get("finalize")
            if finalize:
                try:
                    value = finalize(task)
                    if asyncio.iscoroutine(value):
                        await value
                except Exception as exc:
                    task.errors.append(f"finalize:{type(exc).__name__}")

    @staticmethod
    def _record_outcome(task: DrawTask, outcome, results: list[ImageResult]) -> bool:
        value = getattr(outcome, "generation_success", None)
        generation_success = bool(results) if value is None else bool(value)
        usable = getattr(outcome, "usable_output_count", None)
        usable_output_count = len(results) if usable is None and generation_success else int(usable or 0)
        delivery_kind = str(
            getattr(outcome, "delivery_kind", "")
            or ("image" if generation_success else "notification")
        )
        task.runtime["generation_success"] = generation_success
        task.runtime["usable_output_count"] = max(0, usable_output_count)
        prefix = "image_delivery" if delivery_kind == "image" else "notification_delivery"
        task.runtime[f"{prefix}_attempted"] = True
        task.runtime[f"{prefix}_success"] = bool(getattr(outcome, "success", outcome))
        task.runtime[f"{prefix}_error"] = str(getattr(outcome, "error", "") or "")
        task.runtime[f"{prefix}_side_effects_started"] = bool(
            getattr(outcome, "side_effects_started", False)
        )
        task.runtime[f"{prefix}_side_send_started"] = bool(
            getattr(outcome, "side_send_started", False)
        )
        task.runtime[f"{prefix}_side_send_error"] = str(
            getattr(outcome, "side_send_error", "") or ""
        )
        if not generation_success and results and "no_usable_outputs" not in task.errors:
            task.errors.append("no_usable_outputs")
        return generation_success

    async def _send_terminal_notice(self, task: DrawTask) -> None:
        task.runtime["runner_started"] = True
        try:
            outcome = await self.runner(task, [])
            self._record_outcome(task, outcome, [])
            if not bool(getattr(outcome, "success", outcome)):
                error = str(getattr(outcome, "error", "") or "")
                task.errors.append(f"terminal_notice:{error or 'SendFailed'}")
        except Exception as exc:
            task.errors.append(f"terminal_notice:{type(exc).__name__}")

    async def _generate(self, task: DrawTask) -> list[ImageResult]:
        providers = list(self.config_getter().providers)
        if not providers: raise ConfigurationError("未配置有效图片节点")
        primary_id = str(task.runtime.get("primary_provider_id", ""))
        if primary_id:
            primary = next((item for item in providers if item.id == primary_id), None)
            if primary is not None:
                providers = [primary, *(item for item in providers if item.id != primary_id)]
        last_error = None
        no_output_error = None
        for provider in providers:
            models = []
            for model in (provider.model, *provider.available_models):
                if model not in models:
                    models.append(model)
            if not models:
                models = [""]
            for model_index, model in enumerate(models, start=1):
                attempt = replace(provider, model=model)
                attempt_request = self._request_for_attempt(task, attempt)
                task.runtime["current_provider_id"] = attempt.id
                task.runtime["current_model"] = model
                self.set_stage(task, TaskStage.REQUESTING_PROVIDER)
                adapter = ADAPTERS[attempt.api_type](attempt)
                key_index = self._key_indices.get(attempt.id, 0)
                key = attempt.api_keys[key_index % len(attempt.api_keys)]
                self._key_indices[attempt.id] = key_index + 1
                try:
                    self.debug(
                        task,
                        "task=%s 绘图节点尝试 node=%s type=%s model=%s model_attempt=%d/%d size=%s aspect=%s count=%d references=%d persona_refs=%d",
                        task.id[:8], attempt.id, attempt.api_type, model,
                        model_index, len(models),
                        attempt_request.size or attempt.default_size, attempt_request.aspect_ratio,
                        attempt_request.count, len(attempt_request.references),
                        task.runtime.get("current_persona_reference_count", 0),
                    )
                    async with asyncio.timeout(attempt.timeout):
                        results = await adapter.generate(self.session, attempt_request, key)
                    self.debug(
                        task,
                        "task=%s 绘图模型成功 node=%s model=%s results=%d references=%d persona_refs=%d forms=%s bytes=%d",
                        task.id[:8], attempt.id, model, len(results),
                        len(attempt_request.references),
                        task.runtime.get("current_persona_reference_count", 0),
                        ["bytes" if item.data is not None else "local" if item.local_path else "url" for item in results],
                        sum(len(item.data or b"") for item in results),
                    )
                    return results
                except (ProviderError, asyncio.TimeoutError) as exc:
                    last_error = exc
                    if isinstance(exc, NoOutputError):
                        no_output_error = exc
                    task.runtime.setdefault("attempt_errors", []).append(f"{attempt.id}:{model}:{type(exc).__name__}")
                    self.debug(
                        task,
                        "task=%s 绘图模型失败 node=%s model=%s references=%d persona_refs=%d error_type=%s detail=%s",
                        task.id[:8], attempt.id, model,
                        len(attempt_request.references),
                        task.runtime.get("current_persona_reference_count", 0),
                        type(exc).__name__, str(exc),
                    )
                    self.logger(attempt, exc, task)
                except _AiohttpClientError as exc:
                    wrapped = ProviderError(f"网络请求失败 ({type(exc).__name__})")
                    last_error = wrapped
                    task.runtime.setdefault("attempt_errors", []).append(
                        f"{attempt.id}:{model}:{type(wrapped).__name__}"
                    )
                    self.debug(
                        task,
                        "task=%s 绘图模型失败 node=%s model=%s references=%d persona_refs=%d error_type=%s detail=%s",
                        task.id[:8], attempt.id, model,
                        len(attempt_request.references),
                        task.runtime.get("current_persona_reference_count", 0),
                        type(wrapped).__name__, str(wrapped),
                    )
                    self.logger(attempt, wrapped, task)
        # 零输出说明至少一次 Provider 请求已经完成并可能产生费用。
        # 仍走完整 fallback；全部尝试结束后以 no_output 结算，避免误按 failed 退款。
        self.debug(
            task,
            "task=%s 全部节点尝试结束 attempts=%d no_output=%s last_error=%s",
            task.id[:8],
            len(task.runtime.get("attempt_errors", [])),
            bool(no_output_error),
            type(last_error).__name__ if last_error else "-",
        )
        raise no_output_error or last_error or ProviderError("所有节点都失败")

    async def close(self) -> None:
        self.accepting = False
        handles = [handle for _, handle in self.tasks.values()]
        for handle in handles: handle.cancel()
        if handles: await asyncio.gather(*handles, return_exceptions=True)
        self.tasks.clear()
        self.recent.clear()
