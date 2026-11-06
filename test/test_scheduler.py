import asyncio
import time
import unittest
from unittest.mock import patch

from imago.core.models import DrawTask, GenerationRequest, ImageInput, ImageResult, ProviderConfig, RuntimeConfig, TaskStage, TaskState
from imago.services import scheduler as scheduler_module
from imago.services.scheduler import TaskScheduler


class DummySession: pass


class SchedulerTests(unittest.IsolatedAsyncioTestCase):

    async def test_disabled_provider_is_skipped(self):
        # 停用节点仍被调度＝"临时禁用"变成无效开关，请求白白打到它身上（状态可见性为零）
        tried = []

        class Adapter:
            def __init__(self, config): self.config = config

            async def generate(self, session, request, key):
                tried.append(self.config.id)
                return [ImageResult(data=b"image")]

        old = scheduler_module.ADAPTERS.get("custom_endpoint")
        scheduler_module.ADAPTERS["custom_endpoint"] = Adapter
        providers = (
            ProviderConfig("off", "custom_endpoint", "https://example.invalid", ("key",),
                           model="model", enabled=False),
            ProviderConfig("on", "custom_endpoint", "https://example.invalid", ("key",),
                           model="model"),
        )
        scheduler = TaskScheduler(
            lambda: RuntimeConfig(providers=providers, generation_timeout=30),
            DummySession(), lambda *_: None, lambda *_: None,
        )
        task = DrawTask("", "umo", GenerationRequest("x"))
        try:
            results = await scheduler._generate(task)
            self.assertEqual(tried, ["on"])
            self.assertEqual(len(results), 1)
        finally:
            await scheduler.close()
            if old is None: scheduler_module.ADAPTERS.pop("custom_endpoint", None)
            else: scheduler_module.ADAPTERS["custom_endpoint"] = old

    async def test_selected_primary_provider_is_tried_first_and_invalid_falls_back_to_first(self):
        # 主节点选择错误＝用户配置无效或流量走错节点。
        attempts = []

        class Adapter:
            def __init__(self, config): self.config = config
            async def generate(self, session, request, key):
                attempts.append(self.config.id)
                return [ImageResult(data=b"image")]

        old = scheduler_module.ADAPTERS.get("custom_endpoint")
        scheduler_module.ADAPTERS["custom_endpoint"] = Adapter
        providers = (
            ProviderConfig("node_a", "custom_endpoint", "https://example.invalid/a", ("key",), model="model"),
            ProviderConfig("node_b", "custom_endpoint", "https://example.invalid/b", ("key",), model="model"),
        )
        scheduler = TaskScheduler(
            lambda: RuntimeConfig(providers=providers, generation_timeout=30),
            DummySession(), lambda *_: True, lambda *_: None,
        )
        try:
            selected = DrawTask("selected", "umo", GenerationRequest("x"))
            selected.runtime["primary_provider_id"] = "node_b"
            await scheduler._generate(selected)
            invalid = DrawTask("invalid", "umo", GenerationRequest("x"))
            invalid.runtime["primary_provider_id"] = "missing"
            await scheduler._generate(invalid)
            self.assertEqual(attempts, ["node_b", "node_a"])
        finally:
            await scheduler.close()
            if old is None: scheduler_module.ADAPTERS.pop("custom_endpoint", None)
            else: scheduler_module.ADAPTERS["custom_endpoint"] = old


    async def test_persona_references_are_randomly_sampled_per_model_limit(self):
        # Persona 图采样超限或不随机＝参考图分配错误。
        captured = []
        finished = asyncio.Event()
        explicit = ImageInput(b"explicit", "image/png", "explicit.png")
        persona = [
            ImageInput(b"persona-1", "image/png", "p1.png"),
            ImageInput(b"persona-2", "image/png", "p2.png"),
            ImageInput(b"persona-3", "image/png", "p3.png"),
        ]

        class Adapter:
            def __init__(self, config): self.config = config
            async def generate(self, session, request, key):
                captured.append([item.data for item in request.references])
                return [ImageResult(data=b"image")]

        old = scheduler_module.ADAPTERS.get("custom_endpoint")
        scheduler_module.ADAPTERS["custom_endpoint"] = Adapter
        provider = ProviderConfig(
            "node", "custom_endpoint", "https://example.invalid", ("key",),
            model="model", reference_image_limit=3,
        )
        async def runner(task, results): finished.set(); return True
        scheduler = TaskScheduler(
            lambda: RuntimeConfig(providers=(provider,), generation_timeout=30),
            DummySession(), runner, lambda *_: None,
        )
        task = DrawTask("", "umo", GenerationRequest("x", references=[explicit]))
        task.runtime["persona_references"] = persona
        try:
            with patch.object(scheduler_module.random, "sample", return_value=[persona[2], persona[0]]) as sample:
                scheduler.submit(task)
                await asyncio.wait_for(finished.wait(), 1)
            sample.assert_called_once_with(persona, 2)
            self.assertEqual(captured, [[b"explicit", b"persona-3", b"persona-1"]])
            self.assertEqual(task.runtime["current_explicit_reference_count"], 1)
            self.assertEqual(task.runtime["current_persona_reference_count"], 2)
        finally:
            await scheduler.close()
            if old is None: scheduler_module.ADAPTERS.pop("custom_endpoint", None)
            else: scheduler_module.ADAPTERS["custom_endpoint"] = old

    def test_explicit_references_are_kept_when_they_fill_the_limit(self):
        # 用户附图被 Persona 挤掉＝用户请求被静默改写。
        explicit = [
            ImageInput(b"explicit-1", "image/png", "e1.png"),
            ImageInput(b"explicit-2", "image/png", "e2.png"),
        ]
        persona = [ImageInput(b"persona", "image/png", "p.png")]
        provider = ProviderConfig(
            "node", "custom_endpoint", "https://example.invalid", ("key",),
            model="model", reference_image_limit=2,
        )
        task = DrawTask("task", "umo", GenerationRequest("x", references=explicit))
        task.runtime["persona_references"] = persona
        with patch.object(scheduler_module.random, "sample") as sample:
            request = TaskScheduler._request_for_attempt(task, provider)
        sample.assert_not_called()
        self.assertEqual(request.references, explicit)
        self.assertEqual(task.runtime["current_persona_reference_count"], 0)
        # 关系声明按实际采样数量追加：2 张用户图、0 张人设图 → 只有基础声明，
        # 不含"前 N 张/其余 M 张"的分角色段落。
        self.assertIn(scheduler_module.REFERENCE_RELATION_SUFFIX, request.prompt)
        self.assertNotIn("The first", request.prompt)

    def test_persona_only_references_get_identity_suffix(self):
        # 只有人设图时也要追加关系声明：漏掉这条，模型会把参考图当本轮画面源，
        # /拍照 就会照着参考图的背景构图出图（实测出景物）。
        persona = [ImageInput(b"p1", "image/png", "p1.png")]
        provider = ProviderConfig(
            "node", "custom_endpoint", "https://example.invalid", ("key",),
            model="model", reference_image_limit=1,
        )
        task = DrawTask("task", "umo", GenerationRequest("x"))
        task.runtime["persona_references"] = persona
        request = TaskScheduler._request_for_attempt(task, provider)
        self.assertEqual(len(request.references), 1)
        self.assertIn(scheduler_module.REFERENCE_SUFFIX_MARKER, request.prompt)


    async def test_reference_prepare_runs_in_task_before_provider(self):
        # 取图未完成就调 Provider＝参考图缺失静默退化。
        prepare_started = asyncio.Event()
        allow_prepare = asyncio.Event()
        provider_called = asyncio.Event()

        class Adapter:
            def __init__(self, config): self.config = config
            async def generate(self, session, request, key):
                self_test.assertEqual(len(request.references), 1)
                provider_called.set()
                return [ImageResult(data=b"result")]

        self_test = self
        old = scheduler_module.ADAPTERS.get("custom_endpoint")
        scheduler_module.ADAPTERS["custom_endpoint"] = Adapter
        provider = ProviderConfig("node", "custom_endpoint", "https://example.invalid", ("key",), model="model")
        async def runner(task, results): return True
        scheduler = TaskScheduler(lambda: RuntimeConfig(providers=(provider,), generation_timeout=30), DummySession(), runner, lambda *_: None)
        task = DrawTask("", "umo", GenerationRequest("x"))

        async def prepare(current):
            prepare_started.set()
            await allow_prepare.wait()
            from imago.core.models import ImageInput
            current.request.references.append(ImageInput(b"reference", "image/png"))

        task.runtime["prepare"] = prepare
        try:
            task_id = scheduler.submit(task)
            self.assertTrue(task_id)
            await asyncio.wait_for(prepare_started.wait(), 1)
            self.assertFalse(provider_called.is_set())
            allow_prepare.set()
            await asyncio.wait_for(provider_called.wait(), 1)
        finally:
            await scheduler.close()
            if old is None: scheduler_module.ADAPTERS.pop("custom_endpoint", None)
            else: scheduler_module.ADAPTERS["custom_endpoint"] = old

    async def test_no_provider_fails_and_reports(self):
        # 无有效节点不报失败＝任务悬挂。
        finished = asyncio.Event()
        async def runner(task, results): finished.set(); return False
        scheduler = TaskScheduler(lambda: RuntimeConfig(providers=(), generation_timeout=30), DummySession(), runner, lambda *_: None)
        task = DrawTask("", "umo", GenerationRequest("x"))
        scheduler.submit(task)
        await asyncio.wait_for(finished.wait(), 1)
        self.assertEqual(task.state, TaskState.FAILED)
        # 失败原因写入 runtime，供失败配文透传给 LLM 转述。
        self.assertTrue(task.runtime.get("last_provider_error"))
        await scheduler.close()

    async def test_terminate_cancels_waiting_task_without_output(self):
        # 关停不取消＝任务悬挂/资源泄漏。
        output = []
        finalized = []
        async def runner(task, results): output.append(task.state); return True
        scheduler = TaskScheduler(lambda: RuntimeConfig(providers=(), generation_timeout=30), DummySession(), runner, lambda *_: None)
        task = DrawTask("", "umo", GenerationRequest("x"))
        task.runtime["prepare"] = lambda _: asyncio.sleep(10)

        async def finalize(current):
            await asyncio.sleep(0)
            finalized.append((current.state, current.stage))

        task.runtime["finalize"] = finalize
        scheduler.submit(task)
        await asyncio.sleep(0)
        await scheduler.close()
        self.assertEqual(task.state, TaskState.CANCELLED)
        self.assertEqual(task.stage, TaskStage.FINISHED)
        self.assertEqual(output, [])
        self.assertEqual(finalized, [(TaskState.CANCELLED, TaskStage.FINISHED)])

    async def test_terminal_notice_error_still_archives_finished_state(self):
        # 通知失败不归档＝终态丢失、无法复查。
        async def runner(task, results):
            raise RuntimeError("send failed")

        scheduler = TaskScheduler(lambda: RuntimeConfig(providers=(), generation_timeout=30), DummySession(), runner, lambda *_: None)
        task = DrawTask("", "umo", GenerationRequest("x"))
        scheduler.submit(task)
        for _ in range(20):
            await asyncio.sleep(0)
            if task.stage == TaskStage.FINISHED:
                break
        self.assertEqual(task.state, TaskState.FAILED)
        self.assertEqual(task.stage, TaskStage.FINISHED)
        self.assertTrue(any(value.startswith("terminal_notice:") for value in task.errors))
        await scheduler.close()

    async def test_prequeue_reference_time_counts_toward_total_timeout(self):
        # 排队时间不计入超时＝总时长无限。
        provider_called = False
        finished = asyncio.Event()

        class Adapter:
            def __init__(self, config): self.config = config
            async def generate(self, session, request, key):
                nonlocal provider_called
                provider_called = True
                return [ImageResult(data=b"image")]

        old = scheduler_module.ADAPTERS.get("custom_endpoint")
        scheduler_module.ADAPTERS["custom_endpoint"] = Adapter
        provider = ProviderConfig("node", "custom_endpoint", "https://example.invalid", ("key",), model="model")
        async def runner(task, results): finished.set(); return False
        scheduler = TaskScheduler(lambda: RuntimeConfig(providers=(provider,), generation_timeout=30), DummySession(), runner, lambda *_: None)
        task = DrawTask("", "umo", GenerationRequest("x"), created_at=time.monotonic() - 31)
        try:
            scheduler.submit(task)
            await asyncio.wait_for(finished.wait(), 1)
            self.assertEqual(task.state, TaskState.TIMED_OUT)
            self.assertFalse(provider_called)
        finally:
            await scheduler.close()
            if old is None: scheduler_module.ADAPTERS.pop("custom_endpoint", None)
            else: scheduler_module.ADAPTERS["custom_endpoint"] = old

    async def test_runner_exception_after_generation_is_delivery_failure(self):
        # runner 抛异常时若按 FAILED 记：既退款又按"生成失败"配文，已产出的图等于被吞。
        finished = asyncio.Event()

        class Adapter:
            def __init__(self, config): self.config = config

            async def generate(self, session, request, key):
                return [ImageResult(data=b"image")]

        old = scheduler_module.ADAPTERS.get("custom_endpoint")
        scheduler_module.ADAPTERS["custom_endpoint"] = Adapter
        provider = ProviderConfig("node", "custom_endpoint", "https://example.invalid", ("key",), model="model")

        async def runner(task, results):
            finished.set()
            raise RuntimeError("send boom")

        scheduler = TaskScheduler(
            lambda: RuntimeConfig(providers=(provider,), generation_timeout=30),
            DummySession(), runner, lambda *_: None,
        )
        task = DrawTask("", "umo", GenerationRequest("x"))
        try:
            scheduler.submit(task)
            await asyncio.wait_for(finished.wait(), 1)
            await asyncio.sleep(0)
            self.assertEqual(task.state, TaskState.DELIVERY_FAILED)
            self.assertTrue(task.runtime.get("generation_success"))
        finally:
            await scheduler.close()
            if old is None: scheduler_module.ADAPTERS.pop("custom_endpoint", None)
            else: scheduler_module.ADAPTERS["custom_endpoint"] = old

    async def test_generated_image_send_failure_is_delivery_failure_without_second_notice(self):
        # 发送失败记成生成失败＝状态误报。
        calls = []
        finished = asyncio.Event()

        class Adapter:
            def __init__(self, config): self.config = config
            async def generate(self, session, request, key):
                return [ImageResult(data=b"image")]

        class Outcome:
            success = False
            error = "SendError"

        old = scheduler_module.ADAPTERS.get("custom_endpoint")
        scheduler_module.ADAPTERS["custom_endpoint"] = Adapter
        provider = ProviderConfig("node", "custom_endpoint", "https://example.invalid", ("key",), model="model")
        async def runner(task, results):
            calls.append(list(results))
            finished.set()
            return Outcome()
        scheduler = TaskScheduler(lambda: RuntimeConfig(providers=(provider,), generation_timeout=30), DummySession(), runner, lambda *_: None)
        task = DrawTask("", "umo", GenerationRequest("x"))
        try:
            scheduler.submit(task)
            await asyncio.wait_for(finished.wait(), 1)
            await asyncio.sleep(0)
            self.assertEqual(task.state, TaskState.DELIVERY_FAILED)
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(calls[0]), 1)
            self.assertIn("delivery:SendError", task.errors)
        finally:
            await scheduler.close()
            if old is None: scheduler_module.ADAPTERS.pop("custom_endpoint", None)
            else: scheduler_module.ADAPTERS["custom_endpoint"] = old

    async def test_zero_usable_images_has_no_output_state_even_when_notification_succeeds(self):
        # 0 张图记成功＝空结果当成功。
        finished = asyncio.Event()

        class Adapter:
            def __init__(self, config): self.config = config
            async def generate(self, session, request, key):
                return [ImageResult(data=b"provider-result")]

        class Outcome:
            success = True
            error = ""
            generation_success = False
            usable_output_count = 0
            delivery_kind = "notification"
            side_effects_started = True
            side_send_started = False
            side_send_error = ""

        old = scheduler_module.ADAPTERS.get("custom_endpoint")
        scheduler_module.ADAPTERS["custom_endpoint"] = Adapter
        provider = ProviderConfig("node", "custom_endpoint", "https://example.invalid", ("key",), model="model")

        async def runner(task, results):
            finished.set()
            return Outcome()

        scheduler = TaskScheduler(
            lambda: RuntimeConfig(providers=(provider,), generation_timeout=30),
            DummySession(), runner, lambda *_: None,
        )
        task = DrawTask("", "umo", GenerationRequest("x"))
        try:
            scheduler.submit(task)
            await asyncio.wait_for(finished.wait(), 1)
            await asyncio.sleep(0)
            self.assertEqual(task.state, TaskState.NO_OUTPUT)
            self.assertFalse(task.runtime["generation_success"])
            self.assertEqual(task.runtime["usable_output_count"], 0)
            self.assertTrue(task.runtime["notification_delivery_success"])
            self.assertNotIn("image_delivery_success", task.runtime)
        finally:
            await scheduler.close()
            if old is None: scheduler_module.ADAPTERS.pop("custom_endpoint", None)
            else: scheduler_module.ADAPTERS["custom_endpoint"] = old

    async def test_network_error_falls_back_but_program_error_fails_fast(self):
        # 程序错误被当网络错误重试＝掩盖 bug 并重复扣额。
        attempts = []

        class NetworkAdapter:
            def __init__(self, config): self.config = config
            async def generate(self, session, request, key):
                attempts.append(self.config.model)
                if self.config.model == "network_bad":
                    raise scheduler_module._AiohttpClientError("offline")
                return [ImageResult(data=b"image")]

        old = scheduler_module.ADAPTERS.get("custom_endpoint")
        scheduler_module.ADAPTERS["custom_endpoint"] = NetworkAdapter
        provider = ProviderConfig(
            "node", "custom_endpoint", "https://example.invalid", ("key",),
            model="network_bad", available_models=("ok",),
        )
        scheduler = TaskScheduler(
            lambda: RuntimeConfig(providers=(provider,), generation_timeout=30),
            DummySession(), lambda *_: True, lambda *_: None,
        )
        try:
            network_task = DrawTask("network", "umo", GenerationRequest("x"))
            results = await scheduler._generate(network_task)
            self.assertEqual(len(results), 1)
            self.assertEqual(attempts, ["network_bad", "ok"])

            attempts.clear()

            class ProgramAdapter:
                def __init__(self, config): self.config = config
                async def generate(self, session, request, key):
                    attempts.append(self.config.model)
                    if self.config.model == "buggy":
                        raise TypeError("adapter bug")
                    return [ImageResult(data=b"image")]

            scheduler_module.ADAPTERS["custom_endpoint"] = ProgramAdapter
            buggy = ProviderConfig(
                "node", "custom_endpoint", "https://example.invalid", ("key",),
                model="buggy", available_models=("must_not_run",),
            )
            scheduler.config_getter = lambda: RuntimeConfig(providers=(buggy,), generation_timeout=30)
            with self.assertRaisesRegex(TypeError, "adapter bug"):
                await scheduler._generate(DrawTask("program", "umo", GenerationRequest("x")))
            self.assertEqual(attempts, ["buggy"])
        finally:
            await scheduler.close()
            if old is None: scheduler_module.ADAPTERS.pop("custom_endpoint", None)
            else: scheduler_module.ADAPTERS["custom_endpoint"] = old



if __name__ == "__main__": unittest.main()
