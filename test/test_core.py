import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from imago.core.config import load_config, persona_provider_settings
from imago.core.errors import (
    DuplicateImage,
    NoOutputError,
    ProviderError,
    UnsupportedResponse,
    safe_creation_error_message,
)
from imago.core.models import GenerationRequest, ImageInput, ProviderConfig, QuotaConfig, TaskState
from imago.providers.base import ProviderAdapter
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


class ConfigTests(unittest.TestCase):
    def test_defaults_and_bounds(self):
        cfg = load_config({"task_config": {"generation_timeout": 1, "max_concurrent_tasks": 0}})
        self.assertEqual(cfg.generation_timeout, 30)
        self.assertEqual(cfg.max_concurrent_tasks, 1)

    def test_persona_provider_settings_prefers_umo(self):
        # 会话命中配置与默认配置的 default_personality 不同时，必须用会话命中
        # 配置（与主链 _decorate_llm_request 一致），不得用默认配置覆盖。
        umo = {"provider_settings": {"default_personality": "gpt_demo"}}
        default = {"provider_settings": {"default_personality": "persona_demo"}}
        self.assertEqual(
            persona_provider_settings(umo, default),
            {"default_personality": "gpt_demo"},
        )

    def test_persona_provider_settings_falls_back_to_default(self):
        # 未取到会话命中配置时才回退默认配置。
        self.assertEqual(
            persona_provider_settings(None, {"provider_settings": {"default_personality": "persona_demo"}}),
            {"default_personality": "persona_demo"},
        )

    def test_persona_provider_settings_empty_umo_keeps_empty(self):
        # 会话命中配置存在但 provider_settings 为空时返回空 dict（主链语义：
        # 不吞回全局默认，default_personality 缺失由 resolve 走 conversation
        # persona 分支）。
        self.assertEqual(
            persona_provider_settings({"other": 1}, {"provider_settings": {"default_personality": "persona_demo"}}),
            {},
        )

    def test_extra_params_reject_reserved_keys(self):
        # 保留键必须报错拒绝，防止覆盖 n/model/size/prompt/messages 放大成本
        # 或覆盖节点配置。
        for key in ("n", "model", "size", "prompt", "count", "messages", "api_key", "timeout"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError) as ctx:
                    parse_extra_params(f"--{key} value")
                self.assertIn(f"不允许的附加参数: {key}", str(ctx.exception))

    def test_optimizer_style_labels_are_normalized(self):
        expected = {
            "None(无)": "none",
            "default(通用)": "default",
            "realistic(写实)": "realistic",
            "cinematic(电影感)": "cinematic",
            "anime(动漫)": "anime",
            "3d(3D渲染)": "3d",
            "realistic": "realistic",
        }
        for configured, internal in expected.items():
            with self.subTest(configured=configured):
                cfg = load_config({"optimizer_config": {"optimizer_style": configured}})
                self.assertEqual(cfg.optimizer_style, internal)

        invalid = load_config({"optimizer_config": {"optimizer_style": "unknown"}})
        self.assertEqual(invalid.optimizer_style, "default")

    def test_llm_retry_default_and_bounds(self):
        self.assertEqual(load_config({}).llm_retry, 1)
        self.assertEqual(load_config({"task_config": {"llm_retry": 0}}).llm_retry, 1)
        self.assertEqual(load_config({"task_config": {"llm_retry": 9}}).llm_retry, 5)
        self.assertEqual(load_config({"task_config": {"llm_retry": 3}}).llm_retry, 3)

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

    def test_visual_summary_invalidates_when_selected_reference_disappears(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PersonaStore(Path(tmp), 1024)
            reference = store.add_reference("p", b"new-image", "image/png")
            store.set_summary("p", "prompt", "summary", manual=False, reference_names=[reference["name"]])
            self.assertIsNotNone(store.get_summary("p", "prompt"))
            store.delete_reference("p", reference["name"])
            self.assertIsNone(store.get_summary("p", "prompt"))

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
    def test_failed_and_cancelled_states_are_refundable(self):
        for state in TaskState:
            expected = 3 if state in (TaskState.FAILED, TaskState.CANCELLED) else 0
            self.assertEqual(terminal_refund_amount(state, 3), expected, state.value)
        self.assertEqual(terminal_refund_amount(TaskState.FAILED, 0), 0)
        self.assertEqual(terminal_refund_amount(TaskState.CANCELLED, 0), 0)

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

    def test_gemini_file_uri_appends_key_when_missing(self):
        from imago.providers.gemini import _file_uri_with_key
        self.assertEqual(
            _file_uri_with_key("https://example.invalid/v1beta/files/abc", "k1"),
            "https://example.invalid/v1beta/files/abc?key=k1",
        )
        self.assertEqual(
            _file_uri_with_key("https://example.invalid/v1beta/files/abc?alt=media", "k1"),
            "https://example.invalid/v1beta/files/abc?alt=media&key=k1",
        )
        # 已带 key 或 api_key 为空时原样返回
        self.assertEqual(
            _file_uri_with_key("https://example.invalid/v1beta/files/abc?key=old", "k1"),
            "https://example.invalid/v1beta/files/abc?key=old",
        )
        self.assertEqual(
            _file_uri_with_key("https://example.invalid/v1beta/files/abc", ""),
            "https://example.invalid/v1beta/files/abc",
        )

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

class PromptingTests(unittest.TestCase):
    def test_optimizer_uses_safe_default_and_preserves_user_priority(self):
        prompt = optimizer_system("", "anime", persona=True)
        self.assertIn(DEFAULT_OPTIMIZER_SYSTEM, prompt)
        self.assertIn("始终优先", prompt)
        self.assertIn("不得复制", prompt)
        self.assertIn("风格预设：", prompt)

class SafeCreationErrorTests(unittest.TestCase):
    def test_whitelist_messages_pass_through(self):
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
                self.assertEqual(safe_creation_error_message(ValueError(text)), text)
        # 冒号结尾的前缀允许后跟参数
        self.assertEqual(safe_creation_error_message(ValueError("绘图额度不足: 3")), "绘图额度不足: 3")
        self.assertEqual(safe_creation_error_message(ValueError("不允许的附加参数: n")), "不允许的附加参数: n")

    def test_reference_messages_exact_only_and_http_status(self):
        for text in ("参考图过大", "远程响应不是图片", "不允许访问私网或本地地址", "图片格式或大小不符合要求"):
            with self.subTest(text=text):
                self.assertEqual(safe_creation_error_message(ValueError(text)), text)
        # 拼接变体与三位数以外的状态码不放行
        self.assertEqual(safe_creation_error_message(ValueError("参考图过大 extra junk")), "任务参数无效")
        self.assertEqual(safe_creation_error_message(ValueError("参考图 HTTP 502")), "参考图 HTTP 502")
        self.assertEqual(safe_creation_error_message(ValueError("参考图 HTTP 5020")), "任务参数无效")

    def test_unknown_errors_are_generic_and_redacted(self):
        self.assertEqual(safe_creation_error_message(RuntimeError("boom")), "插件暂时无法创建任务")
        message = safe_creation_error_message(RuntimeError("api_key=sk-verysecret"))
        self.assertNotIn("sk-verysecret", message)


class ProviderErrorDetailTests(unittest.IsolatedAsyncioTestCase):
    class _FakeResponse:
        def __init__(self, status=200, payload=None, text=None):
            self.status = status
            self._payload = payload
            self._text = text

        async def json(self, content_type=None):
            if self._payload is None:
                raise ValueError("响应体不是 JSON")
            return self._payload

        async def text(self):
            return self._text or ""

    async def test_nested_openai_error_is_surfaced(self):
        payload = {
            "error": {
                "message": "图片尺寸超限",
                "type": "invalid_request_error",
                "code": "invalid_request",
            }
        }
        with self.assertRaises(ProviderError) as ctx:
            await ProviderAdapter.response_json(self._FakeResponse(status=400, payload=payload))
        message = str(ctx.exception)
        self.assertIn("图片尺寸超限", message)
        self.assertIn("code=invalid_request", message)
        self.assertIn("type=invalid_request_error", message)

    async def test_top_level_code_message_is_surfaced(self):
        payload = {"code": "InvalidParameter", "message": "参数错误"}
        with self.assertRaises(ProviderError) as ctx:
            await ProviderAdapter.response_json(self._FakeResponse(status=400, payload=payload))
        message = str(ctx.exception)
        self.assertIn("code=InvalidParameter", message)
        self.assertIn("参数错误", message)

    async def test_non_json_body_falls_back_to_raw_text(self):
        with self.assertRaises(ProviderError) as ctx:
            await ProviderAdapter.response_json(self._FakeResponse(status=502, text="Gateway Timeout (relay)"))
        message = str(ctx.exception)
        self.assertIn("HTTP 502", message)
        self.assertIn("Gateway Timeout (relay)", message)


if __name__ == "__main__": unittest.main()
