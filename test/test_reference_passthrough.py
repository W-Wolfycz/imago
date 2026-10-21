"""拍照/绘图参考图透传修复的纯逻辑单元测试（不导入 astrbot core）。

覆盖 imago 自身纯模块行为：
- fetch_reference 对 base64:// / data URL / 本地路径的严格解码、大小限制与魔数识别；
- fetch_reference 的 HTTP(S) 重定向 SSRF 加固（IP 字面量，不访问网络）；
- ReferencePlanner 对 Image/引用消息组件的同步本地化、去重与引用 strict 判定；
- OpenAI images/edits 请求的 multipart 字段构造（`image[]`）。

不依赖 AstrBot backend/源码路径，不 mock AstrBot 核心对象；
AstrBot core/Provider/平台/Hook 兼容性由部署端验收。
"""

import asyncio
import base64
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from imago.core.errors import ReferenceImageError
from imago.core.models import GenerationRequest, ImageInput, ProviderConfig
from imago.core.network import detect_image_mime, fetch_reference
from imago.core.references import ReferencePlanner, quoted_reply_missing_images
from imago.providers.custom import CustomEndpointAdapter
from imago.providers.xai_image import XaiImageAdapter
from imago.providers.openai_image import OpenAIImageAdapter

PNG = b"\x89PNG\r\n\x1a\n" + bytes(24)
JPEG = b"\xff\xd8\xff\xe0" + bytes(24)
WEBP = b"RIFF" + bytes(4) + b"WEBP" + bytes(8)
GIF = b"GIF89a" + bytes(16)

def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")

class FetchReferenceBase64Tests(unittest.IsolatedAsyncioTestCase):

    async def test_base64_invalid_or_oversize_raises(self):
        # 超限/非法 base64 不拒绝＝大图悄悄进内存并被上游拒。
        with self.assertRaises(ReferenceImageError):
            await fetch_reference(None, "base64://!!not-base64!!", max_bytes=4096, block_private=True)
        with self.assertRaisesRegex(ReferenceImageError, "过大"):
            await fetch_reference(None, "base64://" + b64(PNG), max_bytes=16, block_private=True)

class Image:
    """带可选 convert_to_file_path 的轻量 Image 组件替身。"""

    def __init__(self, path="", url="", file="", converter=None):
        self.path = path
        self.url = url
        self.file = file
        self._converter = converter

    async def convert_to_file_path(self):
        if self._converter is None:
            raise ValueError("no converter")
        return await self._converter()

class Reply:
    def __init__(self, chain=None, message_str="", id=""):
        self.chain = chain or []
        self.message_str = message_str
        self.id = id

class FakePlain:
    def __init__(self, text=""):
        self.text = text

class FakeResponse:
    """最小响应替身：status / headers / content.iter_chunked()，不访问网络。"""

    def __init__(self, status=200, headers=None, chunks=b""):
        self.status = status
        self.headers = headers or {}
        self._chunks = chunks

    class _Content:
        def __init__(self, chunks):
            self._chunks = chunks

        async def iter_chunked(self, size):
            for offset in range(0, len(self._chunks), size):
                yield self._chunks[offset:offset + size]

    @property
    def content(self):
        return self._Content(self._chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

class FakeRedirectSession:
    """最小会话替身：记录 get 调用并返回预置响应，不访问网络。"""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses[url]

class ReferencePlannerTests(unittest.IsolatedAsyncioTestCase):
    def planner(self, extractor=None, max_upload=1024 * 1024):
        return ReferencePlanner(
            max_upload_bytes=lambda: max_upload,
            extract_quoted_message_images=extractor,
        )

    async def test_remote_url_image_defers_strict_without_converter(self):
        # 远程图未取得却继续生成＝静默退化成纯文生图。
        calls = []

        async def converter():
            calls.append("called")
            return ""

        component = Image(url="https://example.invalid/ref.png", converter=converter)
        local, deferred = await self.planner().plan([component])
        self.assertEqual(local, [])
        self.assertEqual(len(deferred), 1)
        self.assertEqual(deferred[0]["kind"], "source")
        self.assertEqual(deferred[0]["source"], "https://example.invalid/ref.png")
        self.assertTrue(deferred[0]["strict"])
        self.assertEqual(calls, [])
        self.assertNotIn("component", deferred[0])

    async def test_duplicate_images_are_deduplicated(self):
        # 去重失效＝同一张图重复计入参考图上限。
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ref.png"
            path.write_bytes(PNG)
            local, _ = await self.planner().plan([
                Image(path=str(path)),
                Image(path=str(path)),
            ])
            self.assertEqual(len(local), 1)

    async def test_reply_id_only_without_text_is_strict(self):
        """Fix A：只有引用 id、无 chain/message_str 的纯引用必须按 strict 处理。"""
        async def extractor(event, component):
            return []

        reply = Reply(id="msg_quoted_001", chain=[], message_str="")
        local, deferred = await self.planner(extractor=extractor).plan([reply])
        self.assertEqual(local, [])
        self.assertEqual(len(deferred), 1)
        self.assertEqual(deferred[0]["kind"], "reply")
        self.assertTrue(deferred[0]["strict"])

class OpenAIImageEditsFieldTests(unittest.TestCase):
    def test_edits_multipart_uses_image_bracket_field(self):
        # 参考图字段名写错＝上游忽略参考图，仍按文生图出图（静默）。
        captured = []

        class FakeFormData:
            def __init__(self):
                self.fields = []

            def add_field(self, name, value, filename=None, content_type=None):
                self.fields.append((name, value, filename, content_type))

        class FakeAiohttp:
            FormData = FakeFormData

        class Response:
            status = 200

            async def json(self, content_type=None):
                return {"data": [{"url": "https://example.invalid/result.png"}]}

        class RequestContext:
            async def __aenter__(self):
                return Response()

            async def __aexit__(self, *_args):
                return False

        class Session:
            def post(self, url, headers=None, data=None, **kwargs):
                captured.append((url, headers, data, kwargs))
                return RequestContext()

        adapter = OpenAIImageAdapter(ProviderConfig(
            "node", "openai_image", "https://example.invalid/v1", ("key",), model="model",
        ))
        request = GenerationRequest(
            "edit this",
            references=[ImageInput(b"image", "image/png", "ref.png")],
        )
        with patch.dict(sys.modules, {"aiohttp": FakeAiohttp}):
            results = asyncio.run(adapter.generate(Session(), request, "key"))

        self.assertEqual(len(results), 1)
        url, _headers, form, _kwargs = captured[0]
        self.assertTrue(url.endswith("/images/edits"))
        field_names = [item[0] for item in form.fields]
        self.assertIn("image[]", field_names)
        self.assertNotIn("image", field_names)
        image_field = next(item for item in form.fields if item[0] == "image[]")
        self.assertEqual(image_field[1], b"image")
        self.assertEqual(image_field[2], "ref.png")
        self.assertEqual(image_field[3], "image/png")

class ImageSizePassthroughTests(unittest.TestCase):
    """size 只透传：留空时不发送该字段，其它写法原样带上（含比例串与 * 写法）。

    size 现在来自节点「自定义参数」（``custom_params``）或工具显式传入，两者都没有时
    请求里不出现该字段。
    """

    class _Response:
        status = 200

        async def json(self, content_type=None):
            return {"data": [{"url": "https://example.invalid/result.png"}]}

    class _RequestContext:
        async def __aenter__(self): return ImageSizePassthroughTests._Response()

        async def __aexit__(self, *_args): return False

    def _capture(self, *, request_size, node_size, with_references=True):
        captured = []

        class FakeFormData:
            def __init__(self): self.fields = []

            def add_field(self, name, value, filename=None, content_type=None):
                self.fields.append((name, value))

        class Session:
            def post(self, url, headers=None, data=None, json=None, **kwargs):
                captured.append({"url": url, "json": json, "form": data})
                return ImageSizePassthroughTests._RequestContext()

        adapter = OpenAIImageAdapter(ProviderConfig(
            "node", "openai_image", "https://example.invalid/v1", ("key",), model="model",
            custom_params=(("size", node_size),) if node_size else (),
        ))
        references = [ImageInput(b"i", "image/png")] if with_references else []
        with patch.dict(sys.modules, {"aiohttp": type("M", (), {"FormData": FakeFormData})}):
            asyncio.run(adapter.generate(
                Session(), GenerationRequest("draw", size=request_size, references=references), "key",
            ))
        return captured[0]

    def test_empty_size_is_omitted(self):
        # 留空 = 不指定：请求体里不能出现 size（空字符串会被上游当非法参数）
        form_case = self._capture(request_size="", node_size="", with_references=True)
        self.assertTrue(form_case["url"].endswith("/images/edits"))
        self.assertNotIn("size", [name for name, _ in form_case["form"].fields])
        json_case = self._capture(request_size="", node_size="", with_references=False)
        self.assertTrue(json_case["url"].endswith("/images/generations"))
        self.assertNotIn("size", json_case["json"])

    def test_values_pass_through_untouched(self):
        # x 写法、* 写法、比例串、auto、越界值都原样发送，不做任何换算
        for value in ("auto", "1024x1024", "1024*1024", "1:1", "3:4", "2048x2048"):
            with self.subTest(value=value):
                captured = self._capture(request_size=value, node_size="auto")
                self.assertEqual(dict(captured["form"].fields)["size"], value)

    def test_node_size_used_when_request_has_none(self):
        # 节点默认尺寸被忽略＝请求悄悄不带 size。
        captured = self._capture(request_size="", node_size="3:4")
        self.assertEqual(dict(captured["form"].fields)["size"], "3:4")

class CustomAdapterSizeTests(unittest.TestCase):
    def _payload(self, *, request_size, node_size):
        captured = []

        class Response:
            status = 200

            async def json(self, content_type=None):
                return {"data": [{"url": "https://example.invalid/r.png"}]}

        class RequestContext:
            async def __aenter__(self): return Response()

            async def __aexit__(self, *_args): return False

        class Session:
            def post(self, url, headers=None, json=None, **kwargs):
                captured.append(json)
                return RequestContext()

        adapter = CustomEndpointAdapter(ProviderConfig(
            "c", "custom_endpoint", "https://example.invalid", ("k",), model="m",
            custom_params=(("size", node_size),) if node_size else (),
        ))
        asyncio.run(adapter.generate(Session(), GenerationRequest("draw", size=request_size), "k"))
        return captured[0]

class XaiImageAdapterTests(unittest.TestCase):
    """xAI 方言：不注入任何家私有参数、model 原样取配置、改图默认 JSON。

    各家中转的模型 ID 与参数名不同（有的叫 image-2，有的叫 grok-imagine-2.0-ext；
    有的用 quality/nsfw，有的用 providerOptions.nsfw_check），所以插件只保证端点与
    参考图字段名，其余参数一律由节点自定义参数透传。
    """

    class _Response:
        status = 200

        async def json(self, content_type=None):
            return {"data": [{"url": "https://example.invalid/result.png"}]}

    class _RequestContext:
        async def __aenter__(self): return XaiImageAdapterTests._Response()

        async def __aexit__(self, *_args): return False

    def _capture(self, *, with_references, custom_params=(), model="grok-imagine-image-2.0",
                 size="3:4", aspect_ratio="", references=1):
        captured = []

        class FakeFormData:
            def __init__(self): self.fields = []

            def add_field(self, name, value, filename=None, content_type=None):
                self.fields.append((name, value))

        class Session:
            def post(self, url, headers=None, data=None, json=None, **kwargs):
                captured.append({"url": url, "json": json, "form": data, "headers": headers})
                return XaiImageAdapterTests._RequestContext()

        adapter = XaiImageAdapter(ProviderConfig(
            "xai", "xai_image", "https://example.invalid/api/v1", ("key",),
            model=model, custom_params=custom_params,
        ))
        refs = [ImageInput(b"i", "image/png", f"ref-{index}.png") for index in range(references)] if with_references else []
        with patch.dict(sys.modules, {"aiohttp": type("M", (), {"FormData": FakeFormData})}):
            asyncio.run(adapter.generate(
                Session(),
                GenerationRequest("draw", size=size, aspect_ratio=aspect_ratio, references=refs),
                "key",
            ))
        return captured[0]

    def test_generations_pass_model_and_add_no_private_params(self):
        # 插件替服务商塞私有参数＝官方接口可能拒；这条守参数面保持干净。
        payload = self._capture(with_references=False)["json"]
        self.assertEqual(
            (payload["model"], payload["prompt"], payload["n"]),
            ("grok-imagine-image-2.0", "draw", 1),
        )
        # size 与 aspect_ratio 同时发：中转按 size 定尺寸，官方按 aspect_ratio 定比例
        self.assertEqual(payload["size"], "3:4")
        self.assertEqual(payload["aspect_ratio"], "3:4")
        # 插件不替服务商做主：没有配置就不发 quality/resolution 这类家私有字段
        self.assertNotIn("quality", payload)
        self.assertNotIn("resolution", payload)

    def test_ratio_size_is_mirrored_to_aspect_ratio(self):
        # 官方 Grok 方言没有 size，只有 aspect_ratio：比例串要同步过去，两家中转都能跑
        payload = self._capture(with_references=False)["json"]
        self.assertEqual(payload["aspect_ratio"], "3:4")
        pixel = self._capture(with_references=False, size="1024x1024")["json"]
        self.assertEqual(pixel["aspect_ratio"], "1:1")  # 像素尺寸约分成比例
        self.assertEqual(pixel["size"], "1024x1024")
        explicit = self._capture(with_references=False, custom_params=(("aspect_ratio", "16:9"),))["json"]
        self.assertEqual(explicit["aspect_ratio"], "16:9")
        by_request = self._capture(with_references=False, aspect_ratio="9:16")["json"]
        self.assertEqual(by_request["aspect_ratio"], "9:16")

    def test_edits_default_to_json_with_data_url(self):
        # 官方 xAI 只收 JSON：单图放 image{url}，多图放 images[]，url 是 base64 data URL
        case = self._capture(with_references=True)
        self.assertTrue(case["url"].endswith("/images/edits"))
        self.assertEqual(case["headers"]["Content-Type"], "application/json")
        self.assertIsNone(case["form"])
        self.assertEqual(case["json"]["image"]["url"], "data:image/png;base64,aQ==")
        self.assertNotIn("images", case["json"])
        multi = self._capture(with_references=True, references=2)
        self.assertEqual(len(multi["json"]["images"]), 2)
        self.assertNotIn("image", multi["json"])

    def test_multipart_edit_mode_for_relays(self):
        # edit_mode 开关失效或泄漏进请求体＝中转拒绝/参数污染。
        case = self._capture(
            with_references=True, custom_params=(("edit_mode", "multipart"), ("nsfw", "true")),
        )
        names = [name for name, _ in case["form"].fields]
        self.assertNotIn("image[]", names)
        self.assertEqual(names[-1], "image")
        fields = dict(case["form"].fields)
        self.assertEqual(fields["nsfw"], "true")
        self.assertNotIn("edit_mode", fields)  # 本地开关不发给上游

    def test_private_params_pass_through_with_json_types(self):
        # 布尔被当字符串发＝上游校验失败，且只在特定模型暴露。
        payload = self._capture(
            with_references=False,
            custom_params=(("resolution", "2k"), ("quality", "medium"),
                           ("nsfw", "true"), ("providerOptions", '{"nsfw_check": true}')),
        )["json"]
        self.assertEqual((payload["resolution"], payload["quality"]), ("2k", "medium"))
        self.assertEqual((payload["nsfw"], payload["providerOptions"]), (True, {"nsfw_check": True}))
        # 空值表示不发送该字段
        blank = self._capture(with_references=False, custom_params=(("quality", ""),))["json"]
        self.assertNotIn("quality", blank)

class FetchReferenceRedirectTests(unittest.IsolatedAsyncioTestCase):
    """HTTP(S) 重定向 SSRF 加固的行为级测试（IP 字面量，无需 DNS/网络）。"""

    async def test_redirect_is_followed_manually_and_relative_location_resolved(self):
        # 重定向不跟随或相对地址解析错＝远程图取不到。
        responses = {
            "http://1.1.1.1/a.png": FakeResponse(status=302, headers={"Location": "/b.png"}),
            "http://1.1.1.1/b.png": FakeResponse(status=200, headers={"Content-Type": "image/png"}, chunks=PNG),
        }
        session = FakeRedirectSession(responses)
        image = await fetch_reference(session, "http://1.1.1.1/a.png", max_bytes=4096, block_private=True)
        self.assertEqual(image.data, PNG)
        self.assertEqual([url for url, _ in session.calls], ["http://1.1.1.1/a.png", "http://1.1.1.1/b.png"])
        for _url, kwargs in session.calls:
            self.assertIs(kwargs.get("allow_redirects"), False)

    async def test_redirect_to_private_network_is_blocked(self):
        # 重定向到私网未被拦＝SSRF 打内网。
        session = FakeRedirectSession({
            "http://1.1.1.1/a.png": FakeResponse(status=302, headers={"Location": "http://127.0.0.1:8080/ref.png"}),
        })
        with self.assertRaisesRegex(ReferenceImageError, "重定向目标不安全"):
            await fetch_reference(session, "http://1.1.1.1/a.png", max_bytes=4096, block_private=True)
        # 内网目标未被请求（校验在请求前拦截）。
        self.assertEqual([url for url, _ in session.calls], ["http://1.1.1.1/a.png"])

    async def test_redirect_with_embedded_credentials_is_blocked(self):
        # 带凭据的 URL 未被拦＝把密钥发给第三方。
        session = FakeRedirectSession({
            "http://1.1.1.1/a.png": FakeResponse(status=302, headers={"Location": "http://user:pass@1.1.1.1/ref.png"}),
        })
        with self.assertRaises(ReferenceImageError):
            await fetch_reference(session, "http://1.1.1.1/a.png", max_bytes=4096, block_private=True)

    async def test_redirect_limit_exceeded_is_rejected(self):
        # 跳数不设上限＝重定向环无限请求。
        targets = [f"http://1.1.1.1/a{i}.png" for i in range(7)]
        responses = {
            targets[i]: FakeResponse(status=302, headers={"Location": targets[i + 1]})
            for i in range(6)
        }
        session = FakeRedirectSession(responses)
        with self.assertRaisesRegex(ReferenceImageError, "重定向次数过多"):
            await fetch_reference(session, targets[0], max_bytes=4096, block_private=True)
        # 最多 5 跳：请求了 a0..a5，第 6 个重定向被拒绝，a6 不被请求。
        self.assertEqual(len(session.calls), 6)

    async def test_initial_private_url_is_rejected_without_network(self):
        # 初始私网 URL 不拦＝直接打内网。
        session = FakeRedirectSession({})
        with self.assertRaises(ReferenceImageError):
            await fetch_reference(session, "http://127.0.0.1/ref.png", max_bytes=4096, block_private=True)
        self.assertEqual(session.calls, [])

class FetchReferenceHttpContentTests(unittest.IsolatedAsyncioTestCase):
    """HTTP 参考图内容校验：空 body、魔数识别与 verify_magic 开关。"""

    async def test_http_magic_overrides_header_mime(self):
        # MIME 只信响应头＝把 HTML 当图片交给下游。
        session = FakeRedirectSession({
            "http://1.1.1.1/a.gif": FakeResponse(status=200, headers={"Content-Type": "image/gif"}, chunks=JPEG),
        })
        image = await fetch_reference(session, "http://1.1.1.1/a.gif", max_bytes=4096, block_private=True)
        self.assertEqual(image.mime_type, "image/jpeg")
        self.assertEqual(image.data, JPEG)

class FetchReferenceDataUrlTests(unittest.IsolatedAsyncioTestCase):
    """data URL 魔数识别与 base64 解码前长度预估。"""

class ResolveCheckedUrlTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_ip_literal_is_pinned_with_host_header(self):
        # IP 固定丢失＝DNS 重绑定绕过私网校验。
        from imago.core.network import _resolve_checked_url
        request_url, headers = await _resolve_checked_url("http://1.1.1.1/a.png", block_private=True)
        self.assertEqual(request_url, "http://1.1.1.1/a.png")
        self.assertEqual(headers, {"Host": "1.1.1.1"})

class QuotedReplyMissingImagesTests(unittest.TestCase):
    def test_strict_only_fails_when_quoted_content_is_unavailable(self):
        # 有图：不算失败
        self.assertFalse(quoted_reply_missing_images(strict=True, image_sources=["a.png"]))
        # 非严格：不算失败
        self.assertFalse(quoted_reply_missing_images(strict=False, image_sources=[]))
        # 严格 + 无图 + 取到正文：引用消息存在且不含图 → 继续
        self.assertFalse(quoted_reply_missing_images(
            strict=True, image_sources=[], quoted_text="抱歉，这张没拍成"
        ))
        # 严格 + 无图 + 正文只是图片占位符 / 完全取不到：判定失败
        self.assertTrue(quoted_reply_missing_images(
            strict=True, image_sources=[], quoted_text="[图片]"
        ))
        self.assertTrue(quoted_reply_missing_images(strict=True, image_sources=[], quoted_text=""))

if __name__ == "__main__":
    unittest.main()
