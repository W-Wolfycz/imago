import json
import tempfile
import unittest
from pathlib import Path

from imago.core.config import (
    split_custom_params,
    load_config,
    optimizer_in_use,
    persona_provider_settings,
    style_arg_consumed,
)
from imago.core.dedup import DEDUPE_WINDOW_SECONDS, is_duplicate_request
from imago.core.errors import (
    DuplicateImage,
    ProviderError,
    safe_creation_error_message,
)
from imago.core.models import GenerationRequest, ImageInput, ProviderConfig, QuotaConfig, TaskState
from imago.providers.base import ProviderAdapter
from imago.providers.openai_image import OpenAIImageAdapter
from imago.core.prompting import (
    DEFAULT_OPTIMIZER_SYSTEM,
    optimizer_system,
)
from imago.core.security import ensure_child, parse_extra_params, redact, redact_debug, safe_component
from imago.services.persona_store import PersonaStore
from imago.services.quota_store import QuotaStore, terminal_refund_amount

ROOT = Path(__file__).resolve().parents[1]

class ConfigTests(unittest.TestCase):

    def test_reference_limit_and_blank_size_bounds(self):
        # 参考图数量被夹到合法区间（错值会静默改变参考图张数）；尺寸留空是 1.2.1 的
        # 回归点：不能被兜成 1024x1024，否则请求会悄悄带上尺寸
        limited = load_config({"providers": [{
            "id": "node", "api_type": "openai_image", "base_url": "https://example.invalid/v1",
            "api_keys": "key_demo", "reference_image_limit": -5,
        }]})
        self.assertEqual(limited.providers[0].reference_image_limit, 0)
        blank = load_config({"providers": [{
            "id": "node", "api_type": "openai_image", "base_url": "https://example.invalid/v1",
            "api_keys": "key_demo", "custom_params": "size=   \n",
        }]})
        self.assertEqual(blank.providers[0].custom_params, (("size", ""),))

    def test_custom_params_parse_and_legacy_size(self):
        # 自定义参数是唯一入口：写错一行/重复键/保留字段都不能半途炸掉或静默覆盖尺寸。
        # 旧 default_size 必须仍能生效（1.2.x 配置升级后行为不变）。
        cfg = load_config({"providers": [{
            "id": "node", "api_type": "xai_image", "base_url": "https://example.invalid/v1",
            "api_keys": "key_demo",
            "custom_params": "# 注释\nquality=1k\nnsfw: true\nbad line\nsize=3:4\nsize=9:9\n",
        }]})
        self.assertEqual(
            cfg.providers[0].custom_params,
            (("quality", "1k"), ("nsfw", "true"), ("size", "3:4")),
        )
        legacy = load_config({"providers": [{
            "id": "node", "api_type": "openai_image", "base_url": "https://example.invalid/v1",
            "api_keys": "key_demo", "default_size": "1024x1024",
        }]})
        self.assertEqual(legacy.providers[0].custom_params, (("size", "1024x1024"),))
        explicit = load_config({"providers": [{
            "id": "node", "api_type": "openai_image", "base_url": "https://example.invalid/v1",
            "api_keys": "key_demo", "default_size": "1024x1024", "custom_params": "size=1:1\n",
        }]})
        self.assertEqual(explicit.providers[0].custom_params, (("size", "1:1"),))

    def test_persona_provider_settings_uses_umo_config(self):
        # 会话命中配置优先（与主链 _decorate_llm_request 同源），未取到才回退默认；
        # 会话命中配置存在但缺 provider_settings 时返回空 dict，不吞回全局默认。
        umo = {"provider_settings": {"default_personality": "gpt_demo"}}
        default = {"provider_settings": {"default_personality": "persona_demo"}}
        self.assertEqual(persona_provider_settings(umo, default), {"default_personality": "gpt_demo"})
        self.assertEqual(persona_provider_settings(None, default), {"default_personality": "persona_demo"})
        self.assertEqual(persona_provider_settings({"other": 1}, default), {})

    def test_switch_values_are_honored(self):
        # 显式写进配置的开关必须被采纳：配置键被默默忽略属静默失效，看不到报错
        cfg = load_config({
            "optimizer_config": {"optimize_plain_draw": True},
            "task_config": {"at_trigger_user": False},
        })
        self.assertTrue(cfg.optimize_plain_draw)
        self.assertFalse(cfg.at_trigger_user)

    def test_optimizer_in_use_is_independent_of_style_linkage(self):
        # 「是否走副脑」只由总开关 + 任务来源决定，与成像基准的联动无关。
        cfg = load_config({"optimizer_config": {"optimize_plain_draw": True}})
        self.assertTrue(optimizer_in_use(cfg, persona=True, plain_draw_tool=False))
        self.assertTrue(optimizer_in_use(cfg, persona=False, plain_draw_tool=True))
        # /画 等指令路径（plain_draw_tool=False）即使开关开启也不经副脑。
        self.assertFalse(optimizer_in_use(cfg, persona=False, plain_draw_tool=False))
        off = load_config({"optimizer_config": {"optimize_plain_draw": False}})
        self.assertTrue(optimizer_in_use(off, persona=True, plain_draw_tool=False))
        self.assertFalse(optimizer_in_use(off, persona=False, plain_draw_tool=True))
        # 总开关关闭：任何路径都不调用副脑。
        disabled = load_config({"optimizer_config": {"enable_optimizer": False, "optimize_plain_draw": True}})
        self.assertFalse(optimizer_in_use(disabled, persona=True, plain_draw_tool=True))
        self.assertFalse(optimizer_in_use(disabled, persona=False, plain_draw_tool=True))

    def test_style_arg_consumed_only_with_optimizer_or_fallback_suffix(self):
        # 主 LLM 的基准参数只在两种情形被消费：本轮副脑参与，或降级后缀注入开启；
        # 其余情形丢弃（没有消费方）。这是规格边界，改动前请先改这里。
        auto = load_config({"optimizer_config": {"optimizer_style": "auto(自动)"}})
        with_fallback = load_config({"optimizer_config": {
            "optimizer_style": "auto(自动)", "fallback_style_injection": True,
        }})
        self.assertTrue(style_arg_consumed(auto, use_optimizer=True))
        self.assertFalse(style_arg_consumed(auto, use_optimizer=False))
        self.assertTrue(style_arg_consumed(with_fallback, use_optimizer=False))
        self.assertTrue(style_arg_consumed(with_fallback, use_optimizer=True))
        # 配置写死基准时，配置优先与是否消费无关（resolve_style 保证）
        fixed = load_config({"optimizer_config": {"optimizer_style": "realistic(真人实拍)"}})
        self.assertTrue(style_arg_consumed(fixed, use_optimizer=True))

    def test_quota_id_lists_are_line_items_only(self):
        # 名单按行解析；解析错了只会静默改变谁被限流，不会报错
        ids = load_config({"quota_config": {"blacklist_ids": "10001\n10002"}})
        self.assertEqual(ids.quota.blacklist_ids, frozenset({"10001", "10002"}))
        comma = load_config({"quota_config": {"blacklist_ids": "10001,10002"}})
        self.assertEqual(comma.quota.blacklist_ids, frozenset({"10001,10002"}))

class SecurityTests(unittest.TestCase):

    def test_extra_params(self):
        # 附加参数是用户可控入口：保留键被放行会覆盖 n/model/size，放大成本或改配置。
        self.assertEqual(parse_extra_params('--quality high --seed "12"'), {"quality": "high", "seed": "12"})
        with self.assertRaises(ValueError):
            parse_extra_params("--timeout 2")
        # 保留键必须报错拒绝：防止覆盖 n/model/size/prompt/messages 放大成本或改配置。
        for key in ("n", "model", "size", "prompt", "count", "messages"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError) as ctx:
                    parse_extra_params(f"--{key} value")
                self.assertIn(f"不允许的附加参数: {key}", str(ctx.exception))

    def test_redaction(self):
        # 脱敏漏字段会把 key、URL、本地路径写进日志（静默泄漏）。
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
        # 路径校验失效会让 persona 目录外的文件被读写，且不报错。
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(ensure_child(root, root / "a").parent, root.resolve())
            with self.assertRaises(ValueError): ensure_child(root, root / ".." / "escape")
        self.assertNotIn("/", safe_component("../persona"))

class PersonaStoreTests(unittest.TestCase):

    def test_summary_persists_until_explicit_update(self):
        # 摘要只认显式重建/保存；被 prompt 变化冲掉会让外观漂移且无报错。
        with tempfile.TemporaryDirectory() as tmp:
            store = PersonaStore(Path(tmp), 1024)
            store.set_summary("p", "old", "black hair", manual=False)
            # Prompt 变化不再让摘要失效（摘要只认用户显式重建/保存）。
            self.assertEqual(store.get_summary("p", "new")["summary"], "black hair")
            store.set_summary("p", "old", "manual", manual=True)
            self.assertEqual(store.get_summary("p", "new")["summary"], "manual")
            store.add_reference("p", b"image", "image/png")
            with self.assertRaises(DuplicateImage): store.add_reference("p", b"image", "image/png")

    def test_task_inputs_outputs_and_manifest_are_persisted_without_identity(self):
        # 任务清单写入身份字段＝隐私泄漏，且不会报错。
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
        # 失败/取消不退额度＝静默吞掉用户额度。
        for state in TaskState:
            expected = 3 if state in (TaskState.FAILED, TaskState.CANCELLED) else 0
            self.assertEqual(terminal_refund_amount(state, 3), expected, state.value)
        self.assertEqual(terminal_refund_amount(TaskState.FAILED, 0), 0)
        with tempfile.TemporaryDirectory() as tmp:
            policy = QuotaConfig(enabled=True, daily_refresh_enabled=True, daily_quota_target=5)
            store = QuotaStore(Path(tmp), lambda: "2026-07-22")
            self.assertEqual(store.consume("10001", 2, policy).snapshot.quota, 3)
            self.assertEqual(store.refund("10001", 2, policy).quota, 5)

    def test_daily_refresh_resets_low_and_high_balances_to_target(self):
        # 每日刷新漏算某种余额状态＝额度悄悄算错。
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
        # 黑名单用户即使是白名单也要被拒；断言必须区分"黑名单拦下"与"额度不足"，
        # 否则把黑名单逻辑整段架空、只靠 0 额度拒绝也能让 .allowed 为 False（变异验证抓到过）。
        with tempfile.TemporaryDirectory() as tmp:
            policy = QuotaConfig(
                enabled=True,
                daily_quota_target=50,
                blacklist_ids=frozenset({"10001"}),
                unlimited_whitelist_ids=frozenset({"10001", "10002"}),
            )
            store = QuotaStore(Path(tmp), lambda: "2026-07-22")
            check = store.can_consume("10001", 1, policy)
            self.assertFalse(check.allowed)
            self.assertIn("无法使用绘图功能", check.reason)
            # 扣费路径同样要拦：不能只在查询接口里生效
            charged = store.consume("10001", 1, policy)
            self.assertFalse(charged.allowed)
            self.assertEqual(charged.charged, 0)
            self.assertEqual(store.inspect("10001", policy).quota, 50)
            # 白名单用户不受影响且不扣额度
            decision = store.consume("10002", 4, policy)
            self.assertTrue(decision.allowed)
            self.assertEqual(decision.charged, 0)

    def test_checkin_once_per_day_and_bulk_save(self):
        # 签到可重复领奖或批量保存丢失＝静默多发/丢数据。
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

    def test_gemini_file_uri_appends_key_only_for_same_host(self):
        # fileUri 只在同域时补 key；异域对象存储不拼，避免把密钥泄漏给第三方
        from imago.providers.gemini import _file_uri_with_key
        # 同域 Files API：补 key；带 alt 参数时用 &；已带 key 或 key 为空时原样
        self.assertEqual(
            _file_uri_with_key("https://api.example.invalid/v1beta/files/abc", "k1", "api.example.invalid"),
            "https://api.example.invalid/v1beta/files/abc?key=k1",
        )
        self.assertEqual(
            _file_uri_with_key("https://api.example.invalid/f?alt=media", "k1", "api.example.invalid"),
            "https://api.example.invalid/f?alt=media&key=k1",
        )
        self.assertEqual(
            _file_uri_with_key("https://api.example.invalid/f?key=old", "k1", "api.example.invalid"),
            "https://api.example.invalid/f?key=old",
        )
        self.assertEqual(_file_uri_with_key("https://api.example.invalid/f", "", "api.example.invalid"),
                         "https://api.example.invalid/f")
        # 结果图放在第三方对象存储时绝不能把 key 拼过去
        self.assertEqual(_file_uri_with_key("https://cdn.other.invalid/a.jpg", "k1", "api.example.invalid"),
                         "https://cdn.other.invalid/a.jpg")

    def test_typed_param_fields_merge_into_custom_params(self):
        # 配置页控件（size/quality/nsfw）优先于文本；控件留空表示删除该键
        cfg = load_config({"providers": [{
            "id": "n", "api_type": "xai_image", "base_url": "https://example.invalid/v1",
            "api_keys": "k", "model": "m", "size": "3:4", "nsfw": True,
            "custom_params": "quality=medium\nsize=1:1\nextra=1",
        }]})
        params = dict(cfg.providers[0].custom_params)
        self.assertEqual(params["size"], "3:4")
        self.assertEqual(params["nsfw"], "true")
        self.assertEqual((params["quality"], params["extra"]), ("medium", "1"))
        blank = load_config({"providers": [{
            "id": "n", "api_type": "openai_image", "base_url": "https://example.invalid/v1",
            "api_keys": "k", "model": "m", "size": "", "custom_params": "size=1024x1024",
        }]})
        self.assertEqual(blank.providers[0].custom_params, ())

    def test_gemini_ignores_quality_field(self):
        # 配置页对 Gemini 也会写 quality（默认 auto），不能让它在插件页显示成“未识别”
        cfg = load_config({"providers": [{
            "id": "g", "api_type": "gemini_official", "base_url": "https://example.invalid/v1beta",
            "api_keys": "k", "model": "m", "size": "16:9", "quality": "low",
        }]})
        self.assertEqual(dict(cfg.providers[0].custom_params), {"size": "16:9"})
        # custom_params 里手写的键仍然照发（那是用户显式行为）
        explicit = load_config({"providers": [{
            "id": "g", "api_type": "gemini_official", "base_url": "https://example.invalid/v1beta",
            "api_keys": "k", "model": "m", "custom_params": "imageSize=2K",
        }]})
        self.assertEqual(dict(explicit.providers[0].custom_params), {"imageSize": "2K"})

    def test_split_custom_params_respects_allowed_keys(self):
        # 只拆该协议支持的键：其余键必须留在文本里，不能被搬到该协议忽略的字段后丢掉
        known, rest = split_custom_params("size=16:9\nquality=low", ("size",))
        self.assertEqual((known, rest), ({"size": "16:9"}, "quality=low"))
        known, rest = split_custom_params("size=1024x1024\nfoo=1", ())
        self.assertEqual((known, rest), ({}, "size=1024x1024\nfoo=1"))

class PromptingTests(unittest.TestCase):

    def test_optimizer_falls_back_to_safe_default_and_keeps_fixed_protocol(self):
        # 自定义提示词为空时回落到内置安全默认（接线，不校验默认文案内容）
        prompt = optimizer_system("", "illustration", persona=True)
        self.assertIn(DEFAULT_OPTIMIZER_SYSTEM, prompt)

class SafeCreationErrorTests(unittest.TestCase):

    def test_reference_messages_exact_only_and_http_status(self):
        # 错误文案白名单放宽会把内部细节直达用户，或把合法错误吞成通用文案。
        for text in ("参考图过大", "远程响应不是图片", "不允许访问私网或本地地址", "图片格式或大小不符合要求"):
            with self.subTest(text=text):
                self.assertEqual(safe_creation_error_message(ValueError(text)), text)
        # 拼接变体与三位数以外的状态码不放行
        self.assertEqual(safe_creation_error_message(ValueError("参考图过大 extra junk")), "任务参数无效")
        self.assertEqual(safe_creation_error_message(ValueError("参考图 HTTP 502")), "参考图 HTTP 502")
        self.assertEqual(safe_creation_error_message(ValueError("参考图 HTTP 5020")), "任务参数无效")

    def test_unknown_errors_are_generic_and_redacted(self):
        # 未知错误必须通用化并脱敏，否则异常串里的 key 会进用户可见文案。
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
        # 上游嵌套 error 不解析＝用户只看到 HTTP 400，排障信息丢失。
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

    async def test_non_json_body_falls_back_to_raw_text(self):
        # 非 JSON 响应体被丢弃＝排障时看不到中转的真实错误。
        with self.assertRaises(ProviderError) as ctx:
            await ProviderAdapter.response_json(self._FakeResponse(status=502, text="Gateway Timeout (relay)"))
        message = str(ctx.exception)
        self.assertIn("HTTP 502", message)
        self.assertIn("Gateway Timeout (relay)", message)

class DuplicateRequestTests(unittest.TestCase):

    def test_same_trigger_message_is_duplicate_regardless_of_wording(self):
        # 同一条触发消息判重失效＝重复建任务、重复扣额度。
        records = [{"dedupe_key": "A", "trigger_message_id": "m1", "created_at": 100.0}]
        # 同一条触发消息：措辞不同（指纹不同）也判重复——一条消息只建一个任务。
        self.assertTrue(is_duplicate_request(records, dedupe_key="B", trigger_message_id="m1", now=101.0))
        # 用户新发一条消息（ID 不同）：允许再生成同一画面。
        self.assertFalse(is_duplicate_request(records, dedupe_key="A", trigger_message_id="m2", now=101.0))
        # 缺少触发消息 ID：退化为「同内容 + 时间窗」。
        self.assertTrue(is_duplicate_request(records, dedupe_key="A", trigger_message_id="", now=101.0))
        self.assertFalse(is_duplicate_request(records, dedupe_key="B", trigger_message_id="", now=101.0))
        self.assertFalse(
            is_duplicate_request(
                records, dedupe_key="A", trigger_message_id="",
                now=100.0 + DEDUPE_WINDOW_SECONDS + 1,
            )
        )

    def test_in_flight_task_blocks_without_message_id_beyond_window(self):
        # 退化路径：在途任务不受窗口限制（出图可能远超窗口），超窗仍算重复；
        # 已结束任务超过窗口后放行。
        in_flight = [{
            "dedupe_key": "A", "trigger_message_id": "", "created_at": 100.0, "pending": True,
        }]
        self.assertTrue(
            is_duplicate_request(
                in_flight, dedupe_key="A", trigger_message_id="",
                now=100.0 + DEDUPE_WINDOW_SECONDS + 1,
            )
        )
        finished = [{**in_flight[0], "pending": False}]
        self.assertFalse(
            is_duplicate_request(
                finished, dedupe_key="A", trigger_message_id="",
                now=100.0 + DEDUPE_WINDOW_SECONDS + 1,
            )
        )
        # 双方都有消息 ID 时仍只比 ID：在途标记不参与判定。
        self.assertFalse(
            is_duplicate_request(
                [{**in_flight[0], "trigger_message_id": "m1"}],
                dedupe_key="A", trigger_message_id="m2", now=100.0,
            )
        )

if __name__ == "__main__": unittest.main()
