import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from imago.core.config import load_config
from imago.core.errors import DuplicateImage, NoOutputError, ProviderError, UnsupportedResponse
from imago.core.models import DrawTask, GenerationRequest, ImageInput, ProviderConfig, QuotaConfig, TaskState
from imago.providers.dashscope import DashScopeMultimodalAdapter
from imago.providers.openai_chat import OpenAIChatAdapter
from imago.providers.openai_image import OpenAIImageAdapter
from imago.core.prompting import (
    DEFAULT_OPTIMIZER_SYSTEM,
    optimizer_system,
    persona_optimizer_input,
    summary_user_prompt,
)
from imago.core.security import ensure_child, parse_extra_params, redact, redact_debug, safe_component
from imago.services.persona_store import PersonaStore
from imago.services.quota_store import QuotaStore, terminal_refund_amount

ROOT = Path(__file__).resolve().parents[1]


class SourceContractTests(unittest.TestCase):
    def test_photo_command_names_resolved_persona(self):
        source = (ROOT / "main.py").read_text("utf-8")
        self.assertIn("正在为当前人设「{persona}」拍摄", source)
        self.assertNotIn("正在为当前 Persona「{persona}」拍摄", source)
        self.assertIn("PENDING_PHOTO.format(persona=resolved_persona[0])", source)

    def test_debug_trace_does_not_repeat_stable_generation_payload(self):
        main_source = (ROOT / "main.py").read_text("utf-8")
        scheduler_source = (ROOT / "services" / "scheduler.py").read_text("utf-8")
        self.assertNotIn("副脑重构链路", main_source)
        self.assertNotIn("副脑重构结果", main_source)
        self.assertNotIn("外观视觉证据", main_source)
        self.assertIn("最终生成请求", main_source)
        self.assertIn("绘图模型成功", scheduler_source)
        self.assertIn("绘图模型失败", scheduler_source)
        self.assertNotIn("ref_bytes=%d prompt=%s params=%s", scheduler_source)

    def test_quota_tool_filter_and_nested_commands_are_kept(self):
        source = (ROOT / "main.py").read_text("utf-8")
        self.assertIn('req.func_tool.remove_tool("generate_image")', source)
        self.assertIn('req.func_tool.remove_tool("generate_persona_image")', source)
        self.assertNotIn("IMAGO_TASK_STATUS", source)
        self.assertNotIn("IMAGO_ACCESS_STATUS", source)
        self.assertNotIn("extra_user_content_parts", source)
        self.assertNotIn("TextPart", source)
        self.assertIn('task.runtime["quota_charged"] = charged', source)
        self.assertIn("terminal_refund_amount(task.state, charged)", source)
        self.assertIn("self._settle_task_quota(task)", source)
        self.assertIn('@imago_group.group("quota")', source)
        for command in ("add", "del", "set", "show", "sign"):
            self.assertIn(f'@quota_group.command("{command}")', source)


class ConfigTests(unittest.TestCase):
    def test_defaults_and_bounds(self):
        cfg = load_config({"task_config": {"generation_timeout": 1, "max_concurrent_tasks": 0}})
        self.assertEqual(cfg.generation_timeout, 30)
        self.assertEqual(cfg.max_concurrent_tasks, 1)

    def test_invalid_and_duplicate_providers_are_removed(self):
        raw = {"providers": [
            {"id":"a","api_type":"openai_image","base_url":"https://example.invalid/v1","api_keys":"x","timeout":1},
            {"id":"a","api_type":"openai_chat","base_url":"https://example.invalid/v1","api_keys":"y"},
            {"id":"b","api_type":"bad","base_url":"x","api_keys":"y"},
        ]}
        cfg = load_config(raw)
        self.assertEqual([p.id for p in cfg.providers], ["a"])
        self.assertEqual(cfg.providers[0].timeout, 10)

    def test_api_keys_are_line_items_only(self):
        cfg = load_config({"providers": [{
            "id": "node",
            "api_type": "openai_image",
            "base_url": "https://example.invalid/v1",
            "api_keys": "key_a\nkey_b,key_c",
        }]})
        self.assertEqual(cfg.providers[0].api_keys, ("key_a", "key_b,key_c"))

    def test_id_lists_are_line_items_only(self):
        cfg = load_config({"quota_config": {"blacklist_ids": "10001\n10002"}})
        self.assertEqual(cfg.quota.blacklist_ids, frozenset({"10001", "10002"}))
        comma_value = load_config({"quota_config": {"blacklist_ids": "10001,10002"}})
        self.assertEqual(comma_value.quota.blacklist_ids, frozenset({"10001,10002"}))

    def test_api_keys_schema_is_list(self):
        schema = json.loads((ROOT / "_conf_schema.json").read_text("utf-8"))
        item = schema["providers"]["templates"]["provider"]["items"]["api_keys"]
        self.assertEqual(item["type"], "list")

    def test_dashscope_provider_type_is_supported(self):
        cfg = load_config({"providers": [{
            "id":"qwen",
            "api_type":"dashscope_multimodal",
            "base_url":"https://example.invalid/generation",
            "api_keys":"x",
            "model":"qwen-image-3.0-pro",
            "reference_image_limit": 3,
        }]})
        self.assertEqual(cfg.providers[0].api_type, "dashscope_multimodal")
        self.assertEqual(cfg.providers[0].reference_image_limit, 3)

    def test_reference_image_limit_is_clamped(self):
        cfg = load_config({"providers": [{
            "id":"node",
            "api_type":"custom_endpoint",
            "base_url":"https://example.invalid/generation",
            "api_keys":"x",
            "reference_image_limit": -1,
        }]})
        self.assertEqual(cfg.providers[0].reference_image_limit, 0)

    def test_quota_config_bounds_and_id_sets(self):
        cfg = load_config({"quota_config": {
            "enable_quota": True,
            "blacklist_ids": ["10001", "10002"],
            "unlimited_whitelist_ids": "10003\n10004",
            "daily_quota_target": -2,
            "daily_checkin_quota_min": 5,
            "daily_checkin_quota_max": 2,
        }})
        self.assertTrue(cfg.quota.enabled)
        self.assertEqual(cfg.quota.blacklist_ids, frozenset({"10001", "10002"}))
        self.assertEqual(cfg.quota.unlimited_whitelist_ids, frozenset({"10003", "10004"}))
        self.assertEqual(cfg.quota.daily_quota_target, 0)
        self.assertEqual((cfg.quota.checkin_quota_min, cfg.quota.checkin_quota_max), (5, 5))

    def test_legacy_daily_quota_floor_is_not_loaded(self):
        cfg = load_config({"quota_config": {"daily_quota_floor": 9}})
        self.assertEqual(cfg.quota.daily_quota_target, 0)


class SecurityTests(unittest.TestCase):
    def test_extra_params(self):
        self.assertEqual(parse_extra_params('--quality high --seed "12"'), {"quality":"high","seed":"12"})
        with self.assertRaises(ValueError): parse_extra_params("--timeout 2")

    def test_redaction(self):
        text = redact("api_key=secret data:image/png;base64,AAAA")
        self.assertNotIn("secret", text); self.assertNotIn("AAAA", text)
        debug = redact_debug("token=secret https://example.invalid/image?id=123456789 /tmp/private/a.png")
        self.assertNotIn("secret", debug)
        self.assertNotIn("example.invalid", debug)
        self.assertNotIn("123456789", debug)
        self.assertNotIn("/tmp/private", debug)
        long_text = "完整调试内容" * 5000
        self.assertEqual(redact_debug(long_text), long_text)

    def test_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(ensure_child(root, root / "a").parent, root.resolve())
            with self.assertRaises(ValueError): ensure_child(root, root / ".." / "escape")
        self.assertNotIn("/", safe_component("../persona"))


class PersonaStoreTests(unittest.TestCase):
    def test_hash_invalidation_manual_override_and_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PersonaStore(Path(tmp), 1024)
            store.set_summary("p", "old", "black hair", manual=False)
            self.assertIsNone(store.get_summary("p", "new"))
            store.set_summary("p", "old", "manual", manual=True)
            self.assertEqual(store.get_summary("p", "new")["summary"], "manual")
            store.add_reference("p", b"image", "image/png")
            with self.assertRaises(DuplicateImage): store.add_reference("p", b"image", "image/png")

    def test_primary_provider_setting_is_persistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PersonaStore(Path(tmp), 1024)
            self.assertEqual(store.get_primary_provider_id(), "")
            store.set_primary_provider_id("backup_demo")
            self.assertEqual(PersonaStore(Path(tmp), 1024).get_primary_provider_id(), "backup_demo")

    def test_text_summary_ignores_reference_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PersonaStore(Path(tmp), 1024)
            store.set_summary("p", "prompt", "summary", manual=False)
            self.assertIsNotNone(store.get_summary("p", "prompt"))
            store.add_reference("p", b"new-image", "image/png")
            self.assertIsNotNone(store.get_summary("p", "prompt"))

    def test_visual_summary_invalidates_when_selected_reference_disappears(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PersonaStore(Path(tmp), 1024)
            reference = store.add_reference("p", b"new-image", "image/png")
            store.set_summary("p", "prompt", "summary", manual=False, reference_names=[reference["name"]])
            self.assertIsNotNone(store.get_summary("p", "prompt"))
            store.delete_reference("p", reference["name"])
            self.assertIsNone(store.get_summary("p", "prompt"))

    def test_manual_summary_is_not_truncated(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PersonaStore(Path(tmp), 1024)
            summary = "x" * 500
            store.set_summary("p", "prompt", summary, manual=True)
            self.assertEqual(store.get_summary("p", "changed")["summary"], summary)

    def test_upload_limit_getter_is_dynamic(self):
        with tempfile.TemporaryDirectory() as tmp:
            limit = [4]
            store = PersonaStore(Path(tmp), lambda: limit[0])
            with self.assertRaisesRegex(ValueError, "图片格式或大小"):
                store.add_reference("p", b"12345", "image/png")
            limit[0] = 8
            added = store.add_reference("p", b"12345", "image/png")
            self.assertEqual(added["size"], 5)

    def test_task_inputs_outputs_and_manifest_are_persisted_without_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PersonaStore(Path(tmp), 1024)
            task_id = "a" * 32
            inputs = store.persist_task_inputs(task_id, [ImageInput(b"reference", "image/png")])
            output = store.task_output_dir(task_id) / "result.png"
            output.write_bytes(b"result")
            outputs = store.record_task_outputs(task_id, [output])
            store.update_task_manifest(task_id, kind="draw", state="succeeded")
            manifest = json.loads((store.task_dir(task_id) / "task.json").read_text("utf-8"))
            self.assertEqual(inputs[0]["size"], len(b"reference"))
            self.assertEqual(outputs[0]["file"], "result.png")
            self.assertEqual(manifest["input_count"], 1)
            self.assertEqual(manifest["output_count"], 1)
            self.assertFalse(any(key in manifest for key in ("user_id", "bot_id", "umo", "persona_id", "url")))


class QuotaStoreTests(unittest.TestCase):
    def test_only_plain_failed_state_is_refundable(self):
        for state in TaskState:
            expected = 3 if state == TaskState.FAILED else 0
            self.assertEqual(terminal_refund_amount(state, 3), expected, state.value)
        self.assertEqual(terminal_refund_amount(TaskState.FAILED, 0), 0)

    def test_daily_refresh_resets_low_and_high_balances_to_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            today = ["2026-07-22"]
            policy = QuotaConfig(enabled=True, daily_refresh_enabled=True, daily_quota_target=2)
            store = QuotaStore(Path(tmp), lambda: today[0])
            self.assertEqual(store.inspect("10001", policy).quota, 2)
            decision = store.consume("10001", 1, policy)
            self.assertTrue(decision.allowed)
            self.assertEqual(decision.snapshot.quota, 1)
            store.adjust("10001", "add", 5, policy)
            today[0] = "2026-07-23"
            self.assertEqual(store.inspect("10001", policy).quota, 2)
            store.adjust("10001", "set", 0, policy)
            self.assertEqual(store.inspect("10001", policy).quota, 0)
            today[0] = "2026-07-24"
            self.assertEqual(store.inspect("10001", policy).quota, 2)

    def test_blacklist_precedes_unlimited_whitelist(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy = QuotaConfig(
                enabled=True,
                blacklist_ids=frozenset({"10001"}),
                unlimited_whitelist_ids=frozenset({"10001", "10002"}),
            )
            store = QuotaStore(Path(tmp), lambda: "2026-07-22")
            self.assertFalse(store.can_consume("10001", 1, policy).allowed)
            decision = store.consume("10002", 4, policy)
            self.assertTrue(decision.allowed)
            self.assertEqual(decision.charged, 0)

    def test_disabled_quota_and_policy_only_access_do_not_create_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = QuotaStore(root, lambda: "2026-07-22")
            disabled = QuotaConfig(enabled=False, daily_refresh_enabled=True, daily_quota_target=5)
            self.assertTrue(store.can_consume("10001", 1, disabled).allowed)
            self.assertTrue(store.consume("10001", 1, disabled).allowed)
            self.assertFalse((root / "quotas.json").exists())

            blocked = QuotaConfig(enabled=True, blacklist_ids=frozenset({"10002"}))
            self.assertFalse(store.can_consume("10002", 1, blocked).allowed)
            self.assertFalse(store.consume("10002", 1, blocked).allowed)
            self.assertFalse((root / "quotas.json").exists())

    def test_admin_adjustment_rejects_negative_amount(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = QuotaStore(Path(tmp), lambda: "2026-07-22")
            with self.assertRaisesRegex(ValueError, "额度不能小于 0"):
                store.adjust("10001", "set", -1, QuotaConfig())

    def test_refund_restores_actual_charged_amount(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy = QuotaConfig(
                enabled=True,
                daily_refresh_enabled=True,
                daily_quota_target=5,
            )
            store = QuotaStore(Path(tmp), lambda: "2026-07-22")
            self.assertEqual(store.consume("10001", 2, policy).snapshot.quota, 3)
            snapshot = store.refund("10001", 2, policy)
            self.assertEqual(snapshot.quota, 5)

    def test_checkin_once_per_day_and_bulk_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy = QuotaConfig(
                enabled=True,
                daily_refresh_enabled=False,
                checkin_enabled=True,
                checkin_quota_min=2,
                checkin_quota_max=4,
            )
            store = QuotaStore(Path(tmp), lambda: "2026-07-22", randint=lambda low, high: 3)
            first = store.checkin("10001", policy)
            second = store.checkin("10001", policy)
            self.assertTrue(first.success)
            self.assertEqual((first.reward, first.snapshot.quota), (3, 3))
            self.assertFalse(second.success)
            rows = store.set_many([{"user_id": "10001", "quota": 8}, {"user_id": "10002", "quota": 1}], policy)
            self.assertEqual({row["user_id"]: row["quota"] for row in rows}, {"10001": 8, "10002": 1})


class ProviderTests(unittest.TestCase):
    def test_common_response_formats(self):
        adapter = OpenAIImageAdapter(ProviderConfig("a","openai_image","https://example.invalid",("k",)))
        values = adapter.parse_common({"data":[{"url":"https://example.invalid/a.png"},{"b64_json":"aW1hZ2U="}]})
        self.assertEqual(len(values), 2)
        self.assertEqual(values[1].data, b"image")

    def test_valid_responses_without_images_are_no_output(self):
        with self.assertRaises(NoOutputError):
            OpenAIImageAdapter.parse_common({"data": []})
        with self.assertRaises(NoOutputError):
            DashScopeMultimodalAdapter.parse_response({"output": {"choices": []}})

    def test_dashscope_qwen_request_and_response(self):
        adapter = DashScopeMultimodalAdapter(ProviderConfig(
            "qwen", "dashscope_multimodal", "https://example.invalid/generation", ("k",),
            model="qwen-image-3.0-pro",
        ))
        request = GenerationRequest(
            "画一张测试图",
            count=2,
            size="1024x1536",
            references=[ImageInput(b"image", "image/png")],
            extra_params={
                "prompt_extend": "false",
                "negative_prompt": "模糊",
                "seed": "7",
                "watermark": "true",
            },
        )
        body = adapter.build_body(request)
        self.assertEqual(body["model"], "qwen-image-3.0-pro")
        content = body["input"]["messages"][0]["content"]
        self.assertTrue(content[0]["image"].startswith("data:image/png;base64,"))
        self.assertEqual(content[1], {"text": "画一张测试图"})
        self.assertEqual(body["parameters"]["size"], "1024*1536")
        self.assertEqual(body["parameters"]["n"], 2)
        self.assertIs(body["parameters"]["prompt_extend"], False)
        self.assertEqual(body["parameters"]["seed"], 7)
        self.assertIs(body["parameters"]["watermark"], True)
        values = adapter.parse_response({
            "output": {"choices": [{"message": {"content": [{"image": "https://example.invalid/result.png"}]}}]},
        })
        self.assertEqual(values[0].url, "https://example.invalid/result.png")

    def test_http_error_keeps_sanitized_provider_code_and_message(self):
        class Response:
            status = 400

            async def json(self, content_type=None):
                return {
                    "code": "InvalidParameter",
                    "message": "workspace 123456789 rejected https://example.invalid/private",
                }

        adapter = OpenAIImageAdapter(ProviderConfig(
            "a", "openai_image", "https://example.invalid", ("k",)
        ))
        with self.assertRaisesRegex(ProviderError, "HTTP 400 code=InvalidParameter") as raised:
            asyncio.run(adapter.response_json(Response()))
        self.assertNotIn("123456789", str(raised.exception))
        self.assertNotIn("example.invalid", str(raised.exception))

    def test_malformed_chat_payload_is_diagnostic_provider_error(self):
        class Response:
            status = 200

            async def json(self, content_type=None):
                return []

        class RequestContext:
            async def __aenter__(self):
                return Response()

            async def __aexit__(self, *_args):
                return False

        class Session:
            def post(self, *_args, **_kwargs):
                return RequestContext()

        adapter = OpenAIChatAdapter(ProviderConfig(
            "a", "openai_chat", "https://example.invalid/v1", ("k",), model="m"
        ))
        with self.assertRaisesRegex(UnsupportedResponse, "响应格式无效"):
            asyncio.run(adapter.generate(Session(), GenerationRequest("draw"), "key"))

    def test_dashscope_does_not_hardcode_qwen_reference_limits(self):
        adapter = DashScopeMultimodalAdapter(ProviderConfig(
            "qwen", "dashscope_multimodal", "https://example.invalid/generation", ("k",),
            model="qwen-image-2.0-pro",
        ))
        request = GenerationRequest("edit", references=[ImageInput(b"x", "image/png") for _ in range(4)])
        body = adapter.build_body(request)
        content = body["input"]["messages"][0]["content"]
        self.assertEqual(len([item for item in content if "image" in item]), 4)
        source = (ROOT / "providers" / "dashscope.py").read_text("utf-8")
        self.assertNotIn("_QWEN_IMAGE_CAPABILITIES", source)
        self.assertNotIn("max_reference_bytes", source)

    def test_dashscope_success_status_error_keeps_sanitized_code_and_message(self):
        with self.assertRaisesRegex(ProviderError, "code=InvalidParameter") as raised:
            DashScopeMultimodalAdapter.parse_response({
                "code": "InvalidParameter",
                "message": "workspace 123456789 rejected https://example.invalid/private",
            })
        self.assertNotIn("123456789", str(raised.exception))
        self.assertNotIn("example.invalid", str(raised.exception))


class PromptingTests(unittest.TestCase):
    def test_optimizer_uses_safe_default_and_preserves_user_priority(self):
        prompt = optimizer_system("", "anime", persona=True)
        self.assertIn(DEFAULT_OPTIMIZER_SYSTEM, prompt)
        self.assertIn("始终优先", prompt)
        self.assertIn("不得复制", prompt)

    def test_model_inputs_are_delimited(self):
        self.assertEqual(
            persona_optimizer_input("stable", "scene"),
            "<identity_summary>\nstable\n</identity_summary>\n\n<scene_request>\nscene\n</scene_request>",
        )
        summary = summary_user_prompt("persona", "evidence")
        self.assertIn("<persona_prompt>\npersona\n</persona_prompt>", summary)
        self.assertIn("<visual_evidence>\nevidence\n</visual_evidence>", summary)


if __name__ == "__main__": unittest.main()
