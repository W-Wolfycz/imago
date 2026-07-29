import asyncio
import time
import unittest
from unittest.mock import patch

from imago.core.errors import NoOutputError, ProviderError
from imago.core.models import DrawTask, GenerationRequest, ImageInput, ImageResult, ProviderConfig, RuntimeConfig, TaskStage, TaskState
from imago.services import scheduler as scheduler_module
from imago.services.scheduler import TaskScheduler


class DummySession: pass


class SchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_api_keys_rotate_between_attempts(self):
        captured_keys = []

        class Adapter:
            def __init__(self, config): self.config = config
            async def generate(self, session, request, key):
                captured_keys.append(key)
                return [ImageResult(data=b"image")]

        old = scheduler_module.ADAPTERS.get("custom_endpoint")
        scheduler_module.ADAPTERS["custom_endpoint"] = Adapter
        provider = ProviderConfig(
            "node", "custom_endpoint", "https://example.invalid", ("key_a", "key_b"), model="model"
        )
        scheduler = TaskScheduler(
            lambda: RuntimeConfig(providers=(provider,), generation_timeout=30),
            DummySession(), lambda *_: True, lambda *_: None,
        )
        try:
            await scheduler._generate(DrawTask("a", "umo", GenerationRequest("x")))
            await scheduler._generate(DrawTask("b", "umo", GenerationRequest("x")))
            await scheduler._generate(DrawTask("c", "umo", GenerationRequest("x")))
            self.assertEqual(captured_keys, ["key_a", "key_b", "key_a"])
        finally:
            await scheduler.close()
            if old is None: scheduler_module.ADAPTERS.pop("custom_endpoint", None)
            else: scheduler_module.ADAPTERS["custom_endpoint"] = old

    async def test_api_key_rotation_is_independent_per_provider(self):
        captured = []

        class Adapter:
            def __init__(self, config): self.config = config
            async def generate(self, session, request, key):
                captured.append((self.config.id, key))
                return [ImageResult(data=b"image")]

        old = scheduler_module.ADAPTERS.get("custom_endpoint")
        scheduler_module.ADAPTERS["custom_endpoint"] = Adapter
        providers = (
            ProviderConfig("node_a", "custom_endpoint", "https://example.invalid/a", ("a1", "a2"), model="model"),
            ProviderConfig("node_b", "custom_endpoint", "https://example.invalid/b", ("b1", "b2"), model="model"),
        )
        scheduler = TaskScheduler(
            lambda: RuntimeConfig(providers=providers, generation_timeout=30),
            DummySession(), lambda *_: True, lambda *_: None,
        )
        try:
            for provider_id in ("node_a", "node_b", "node_a", "node_b"):
                task = DrawTask(provider_id, "umo", GenerationRequest("x"))
                task.runtime["primary_provider_id"] = provider_id
                await scheduler._generate(task)
            self.assertEqual(captured, [
                ("node_a", "a1"),
                ("node_b", "b1"),
                ("node_a", "a2"),
                ("node_b", "b2"),
            ])
        finally:
            await scheduler.close()
            if old is None: scheduler_module.ADAPTERS.pop("custom_endpoint", None)
            else: scheduler_module.ADAPTERS["custom_endpoint"] = old

    async def test_selected_primary_provider_is_tried_first_and_invalid_falls_back_to_first(self):
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

    async def test_models_fallback_inside_provider_before_next_node(self):
        attempts = []
        debug_calls = []
        finished = asyncio.Event()

        class Adapter:
            def __init__(self, config): self.config = config
            async def generate(self, session, request, key):
                attempts.append((self.config.id, self.config.model))
                if self.config.model != "model_ok":
                    raise ProviderError("failed")
                return [ImageResult(data=b"image")]

        old = scheduler_module.ADAPTERS.get("custom_endpoint")
        scheduler_module.ADAPTERS["custom_endpoint"] = Adapter
        provider = ProviderConfig("node_a", "custom_endpoint", "https://example.invalid", ("key",), model="model_bad", available_models=("model_ok",))
        async def runner(task, results): finished.set(); return True
        scheduler = TaskScheduler(
            lambda: RuntimeConfig(providers=(provider,), generation_timeout=30),
            DummySession(), runner, lambda *_: None,
            lambda task, message, *args: debug_calls.append((task, message, args)),
        )
        task = DrawTask("", "umo", GenerationRequest("x"))
        try:
            scheduler.submit(task)
            await asyncio.wait_for(finished.wait(), 1)
            self.assertEqual(attempts, [("node_a", "model_bad"), ("node_a", "model_ok")])
            attempt_logs = [item for item in debug_calls if "绘图节点尝试" in item[1]]
            self.assertEqual([item[0] for item in attempt_logs], [task, task])
            self.assertEqual([item[2][4:6] for item in attempt_logs], [(1, 2), (2, 2)])
            failure_logs = [item for item in debug_calls if "绘图模型失败" in item[1]]
            success_logs = [item for item in debug_calls if "绘图模型成功" in item[1]]
            self.assertEqual([item[2][2] for item in failure_logs], ["model_bad"])
            self.assertEqual([item[2][2] for item in success_logs], ["model_ok"])
            self.assertEqual(task.state, TaskState.SUCCEEDED)
        finally:
            await scheduler.close()
            if old is None: scheduler_module.ADAPTERS.pop("custom_endpoint", None)
            else: scheduler_module.ADAPTERS["custom_endpoint"] = old

    async def test_persona_references_are_randomly_sampled_per_model_limit(self):
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

    async def test_reference_prepare_runs_in_task_before_provider(self):
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
        finished = asyncio.Event()
        async def runner(task, results): finished.set(); return False
        scheduler = TaskScheduler(lambda: RuntimeConfig(providers=(), generation_timeout=30), DummySession(), runner, lambda *_: None)
        task = DrawTask("", "umo", GenerationRequest("x"))
        scheduler.submit(task)
        await asyncio.wait_for(finished.wait(), 1)
        self.assertEqual(task.state, TaskState.FAILED)
        await scheduler.close()

    async def test_query_is_user_and_bot_scoped_and_keeps_short_terminal_snapshot(self):
        finished = asyncio.Event()
        async def runner(task, results): finished.set(); return False
        scheduler = TaskScheduler(lambda: RuntimeConfig(providers=(), generation_timeout=30), DummySession(), runner, lambda *_: None)
        task = DrawTask("", "platform:GroupMessage:group_demo", GenerationRequest("x"), owner_user_id="10001", bot_instance_id="bot_demo")
        scheduler.submit(task)
        await asyncio.wait_for(finished.wait(), 1)
        await asyncio.sleep(0)
        self.assertEqual(task.stage, TaskStage.FINISHED)
        self.assertEqual(len(scheduler.query(umo=task.umo, owner_user_id="10001", bot_instance_id="bot_demo")), 1)
        self.assertEqual(scheduler.query(umo=task.umo, owner_user_id="10002", bot_instance_id="bot_demo"), [])
        self.assertEqual(scheduler.query(umo=task.umo, owner_user_id="10001", bot_instance_id="bot_other"), [])
        await scheduler.close()

    async def test_terminate_cancels_waiting_task_without_output(self):
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

    async def test_generated_image_send_failure_is_delivery_failure_without_second_notice(self):
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

    async def test_provider_zero_output_falls_back_then_finishes_as_no_output(self):
        attempts = []
        finished = asyncio.Event()

        class Adapter:
            def __init__(self, config): self.config = config
            async def generate(self, session, request, key):
                attempts.append(self.config.model)
                raise NoOutputError("没有图片")

        class Outcome:
            success = True
            error = ""
            generation_success = False
            usable_output_count = 0
            delivery_kind = "notification"

        old = scheduler_module.ADAPTERS.get("custom_endpoint")
        scheduler_module.ADAPTERS["custom_endpoint"] = Adapter
        provider = ProviderConfig(
            "node", "custom_endpoint", "https://example.invalid", ("key",),
            model="empty_a", available_models=("empty_b",),
        )

        async def runner(task, results):
            self.assertEqual(results, [])
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
            self.assertEqual(attempts, ["empty_a", "empty_b"])
            self.assertEqual(task.state, TaskState.NO_OUTPUT)
            self.assertIn("NoOutputError", task.errors)
        finally:
            await scheduler.close()
            if old is None: scheduler_module.ADAPTERS.pop("custom_endpoint", None)
            else: scheduler_module.ADAPTERS["custom_endpoint"] = old

    async def test_network_error_falls_back_but_program_error_fails_fast(self):
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

    async def test_concurrency_limit_increase_and_decrease_are_dynamic(self):
        limit = [1]
        started = {name: asyncio.Event() for name in ("one", "two", "three", "four")}
        gates = {name: asyncio.Event() for name in started}
        active = 0
        peak = 0

        class Adapter:
            def __init__(self, config): self.config = config
            async def generate(self, session, request, key):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                started[request.prompt].set()
                try:
                    await gates[request.prompt].wait()
                    return [ImageResult(data=b"image")]
                finally:
                    active -= 1

        old = scheduler_module.ADAPTERS.get("custom_endpoint")
        scheduler_module.ADAPTERS["custom_endpoint"] = Adapter
        provider = ProviderConfig("node", "custom_endpoint", "https://example.invalid", ("key",), model="model")

        async def runner(task, results):
            return True

        scheduler = TaskScheduler(
            lambda: RuntimeConfig(
                providers=(provider,),
                generation_timeout=30,
                max_concurrent_tasks=limit[0],
            ),
            DummySession(), runner, lambda *_: None,
        )
        tasks = [DrawTask(name, "umo", GenerationRequest(name)) for name in started]
        try:
            scheduler.submit(tasks[0])
            await asyncio.wait_for(started["one"].wait(), 1)
            scheduler.submit(tasks[1])
            await asyncio.sleep(0.05)
            self.assertFalse(started["two"].is_set())

            limit[0] = 2
            await asyncio.wait_for(started["two"].wait(), 1)
            self.assertEqual(peak, 2)

            limit[0] = 1
            scheduler.submit(tasks[2])
            await asyncio.sleep(0.15)
            self.assertFalse(started["three"].is_set())
            gates["one"].set()
            await asyncio.sleep(0.15)
            self.assertFalse(started["three"].is_set())
            gates["two"].set()
            await asyncio.wait_for(started["three"].wait(), 1)
            gates["three"].set()

            scheduler.submit(tasks[3])
            await asyncio.wait_for(started["four"].wait(), 1)
            gates["four"].set()
            for task in tasks:
                for _ in range(30):
                    if task.stage == TaskStage.FINISHED:
                        break
                    await asyncio.sleep(0.01)
                self.assertEqual(task.state, TaskState.SUCCEEDED)
        finally:
            for gate in gates.values():
                gate.set()
            await scheduler.close()
            if old is None: scheduler_module.ADAPTERS.pop("custom_endpoint", None)
            else: scheduler_module.ADAPTERS["custom_endpoint"] = old


if __name__ == "__main__": unittest.main()
