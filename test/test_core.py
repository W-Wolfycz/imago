import ast
import io
import json
import os
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

from imago.core.clock import day_key
from imago.core.dedup import DEDUPE_WINDOW_SECONDS, is_duplicate_request
from imago.core.images import shrink
from imago.core.persona_lookup import (
    allocate_reference_budget,
)
from imago.core.errors import (
    DuplicateImage,
    ProviderError,
    safe_creation_error_message,
)
from imago.core.models import ImageInput, QuotaConfig, TaskState
from imago.providers.base import ProviderAdapter
from imago.core.security import ensure_child, parse_extra_params, redact, redact_debug, safe_component
from imago.services.persona_store import PersonaStore
from imago.services.quota_store import QuotaStore, terminal_refund_amount

ROOT = Path(__file__).resolve().parents[1]

class ClockTests(unittest.TestCase):
    def test_day_key_uses_configured_timezone(self):
        # 时区被忽略＝日切差几小时，额度刷新与每日签到按错的"今天"计算
        self.assertEqual(
            day_key("Asia/Shanghai", datetime(2026, 11, 14, 23, 30, tzinfo=timezone.utc))[0],
            "2026-11-15",
        )
        # 非法时区必须把失败标记出来，否则调用方无从提示
        self.assertTrue(day_key("Not/AZone")[1])

class TaskCacheAndQuotaTests(unittest.TestCase):
    def test_prune_task_cache_keeps_protected_and_newest(self):
        # 缓存永不清理＝磁盘无限增长；protected 被删＝在途任务的输出当场消失
        with tempfile.TemporaryDirectory() as tmp:
            store = PersonaStore(Path(tmp), 16 * 1024 * 1024)
            base = time.time() - 300
            made = []
            for index in range(3):
                task_id = f"{index + 1:032x}"
                outputs = store.task_dir(task_id) / "outputs"
                outputs.mkdir(parents=True, exist_ok=True)
                target = outputs / "a.png"
                target.write_bytes(b"x" * (6 * 1024 * 1024))
                os.utime(target, (base + index, base + index))
                made.append(task_id)
            store.prune_task_cache(16 * 1024 * 1024, protected={made[0]})
            remaining = {path.name for path in (Path(tmp) / "task_cache").iterdir()}
            self.assertIn(made[0], remaining)
            self.assertIn(made[2], remaining)
            self.assertNotIn(made[1], remaining)

    def test_bulk_save_keeps_checkin_state(self):
        # 整表保存抹掉签到日期＝当天能二次签到白拿额度
        with tempfile.TemporaryDirectory() as tmp:
            store = QuotaStore(Path(tmp), lambda: "2026-11-14")
            policy = QuotaConfig(
                enabled=True, checkin_enabled=True, checkin_quota_min=1, checkin_quota_max=1,
            )
            self.assertTrue(store.checkin("10001", policy).success)
            store.set_many([{"user_id": "10001", "quota": 9}], policy)
            self.assertFalse(store.checkin("10001", policy).success)

class PageEndpointContractTests(unittest.TestCase):
    def test_character_endpoints_accept_local_personas(self):
        # 端点用「prompt 非空」判断角色存在 → 本地角色（无人设）传图/存摘要一律 404；
        # 这是页面层才暴露的契约，store 单测和 pyflakes 都看不出来。
        source = (
            Path(__file__).resolve().parents[1] / "integrations" / "web_api.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        lines = source.splitlines()
        bodies = {}
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if isinstance(item, ast.AsyncFunctionDef):
                    bodies[item.name] = "\n".join(lines[item.lineno - 1:item.end_lineno])
        for name in ("detail", "upload", "set_summary"):
            self.assertIn("_character_exists", bodies[name], f"{name} 未用统一存在性判断")

class ReadOnlyAndOverrideTests(unittest.TestCase):
    def test_read_paths_do_not_create_persona_dir(self):
        # 只读接口顺手建目录＝任意 id 都会留下永久垃圾目录，还会让"目录是否存在"的判断自我成立
        with tempfile.TemporaryDirectory() as tmp:
            store = PersonaStore(Path(tmp), 1024)
            self.assertEqual(store.list_references("ghost_persona"), [])
            self.assertNotIn("ghost_persona", store.list_local_names())

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

class ImageShrinkTests(unittest.TestCase):
    def test_oversized_reference_is_downscaled(self):
        # 人设原图常见 5000×5000、单张 10MB+：原样发出去请求体几十 MB，中转截断后
        # 模型只看到边缘，出图跑成景物。这条守的是“超限图必须降采样再发”。
        from PIL import Image

        source = io.BytesIO()
        noise = Image.frombytes("RGB", (2400, 2400), os.urandom(2400 * 2400 * 3))
        noise.save(source, "PNG")
        data, mime = shrink(source.getvalue(), "image/png")
        self.assertEqual(mime, "image/jpeg")
        with Image.open(io.BytesIO(data)) as out:
            self.assertLessEqual(max(out.size), 1536)

    def test_small_reference_is_left_untouched(self):
        # 小图重编码只会白掉画质，还会把 mime 改掉；不该动的时候必须原样返回。
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (96, 96), (30, 60, 90)).save(buffer, "PNG")
        small = buffer.getvalue()
        self.assertEqual(shrink(small, "image/png"), (small, "image/png"))

class PersonaLookupTests(unittest.TestCase):
    def test_allocate_budget_gives_every_character_a_slot(self):
        # 名额被图多的角色占满＝合照少一个人，出图后才发现；每角色至少一张是硬要求。
        self.assertEqual(allocate_reference_budget(3, [8, 8]), [2, 1])
        self.assertEqual(allocate_reference_budget(1, [8, 8]), [1, 0])
        self.assertEqual(allocate_reference_budget(5, [2, 2]), [2, 2])
        self.assertEqual(allocate_reference_budget(0, [8, 8]), [0, 0])

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

    def test_persona_dirs_are_readable_and_migrate_legacy_hash(self):
        # 旧哈希目录要能自动改名成可读名；否则用户根本找不到哪个人设的图
        demo = "persona_demo"
        with tempfile.TemporaryDirectory() as tmp:
            store = PersonaStore(Path(tmp), 1024)
            # 旧目录名是 persona_id 的 UTF-8 十六进制，这里用占位角色名现场算
            legacy = Path(tmp) / "persona_references" / demo.encode("utf-8").hex()
            legacy.mkdir(parents=True)
            (legacy / "a.jpg").write_bytes(b"img")
            refs = store.list_references(demo)
            self.assertEqual([item["name"] for item in refs], ["a.jpg"])
            # 读路径只读不建：旧哈希目录仍在原地，不会被顺手搬家或新建空目录
            self.assertTrue(legacy.is_dir())
            self.assertFalse((Path(tmp) / "persona_references" / demo).exists())
            store.migrate_legacy_dirs()
            readable = Path(tmp) / "persona_references" / demo
            self.assertTrue(readable.is_dir())
            self.assertFalse(legacy.exists())
            marker = json.loads((readable / "persona.json").read_text("utf-8"))
            self.assertEqual(marker["persona_id"], demo)
            # 非法字符替换成下划线；persona.json 不算参考图
            self.assertEqual(store.persona_dir("a/b").name, "a_b")
            self.assertNotIn("persona.json", [item["name"] for item in store.list_references(demo)])

    def test_local_persona_create_is_idempotent_and_delete_clears_summary(self):
        # 重复建要幂等；删除要连摘要一起清——否则重建同名角色会带着上一份旧摘要出图。
        with tempfile.TemporaryDirectory() as tmp:
            store = PersonaStore(Path(tmp), 1024)
            self.assertTrue(store.create_local_persona("测试角色")["created"])
            self.assertFalse(store.create_local_persona("测试角色")["created"])
            # 名字带路径/非法字符时目录名会被改写，角色名与人设/会话对不上，必须直接拒绝
            with self.assertRaises(ValueError):
                store.create_local_persona("a/b")
            self.assertIn("测试角色", store.list_local_names())
            store.set_summary("测试角色", "", "白发少女", manual=True)
            store.delete_local_persona("测试角色")
            self.assertNotIn("测试角色", store.list_local_names())
            self.assertIsNone(store.get_summary("测试角色", ""))

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

class TaskListTests(unittest.TestCase):
    def test_task_output_file_rejects_path_traversal(self):
        # 允许带路径的文件名＝任务页能读到任务目录外的任意文件（路径穿越）。
        with tempfile.TemporaryDirectory() as tmp:
            store = PersonaStore(Path(tmp), 1024)
            task_id = "a" * 32
            outputs = store.task_dir(task_id) / "outputs"
            outputs.mkdir(parents=True, exist_ok=True)
            (outputs / "r.png").write_bytes(b"png")
            self.assertEqual(store.task_output_file(task_id, "r.png").name, "r.png")
            with self.assertRaises(ValueError):
                store.task_output_file(task_id, "../task.json")
            with self.assertRaises(FileNotFoundError):
                store.task_output_file(task_id, "missing.png")

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
