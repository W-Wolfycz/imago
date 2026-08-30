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
from imago.core.references import ReferencePlanner
from imago.providers.openai_image import OpenAIImageAdapter

PNG = b"\x89PNG\r\n\x1a\n" + bytes(24)
JPEG = b"\xff\xd8\xff\xe0" + bytes(24)
WEBP = b"RIFF" + bytes(4) + b"WEBP" + bytes(8)
GIF = b"GIF89a" + bytes(16)

def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")

class FetchReferenceBase64Tests(unittest.IsolatedAsyncioTestCase):
    async def test_base64_url_detects_png(self):
        image = await fetch_reference(None, "base64://" + b64(PNG), max_bytes=4096, block_private=True)
        self.assertEqual(image.mime_type, "image/png")
        self.assertEqual(image.data, PNG)
        self.assertEqual(image.filename, "base64-reference.png")

    async def test_base64_url_detects_jpeg_webp_gif(self):
        cases = (
            (JPEG, "image/jpeg", "base64-reference.jpg"),
            (WEBP, "image/webp", "base64-reference.webp"),
            (GIF, "image/gif", "base64-reference.gif"),
        )
        for payload, expected_mime, expected_name in cases:
            with self.subTest(expected=expected_mime):
                image = await fetch_reference(None, "base64://" + b64(payload), max_bytes=4096, block_private=True)
                self.assertEqual(image.mime_type, expected_mime)
                self.assertEqual(image.data, payload)
                self.assertEqual(image.filename, expected_name)

    async def test_base64_invalid_alphabet_raises(self):
        with self.assertRaises(ReferenceImageError):
            await fetch_reference(None, "base64://!!not-base64!!", max_bytes=4096, block_private=True)

    async def test_base64_empty_payload_raises(self):
        with self.assertRaises(ReferenceImageError):
            await fetch_reference(None, "base64://", max_bytes=4096, block_private=True)

    async def test_base64_unrecognized_magic_raises(self):
        with self.assertRaisesRegex(ReferenceImageError, "无法识别"):
            await fetch_reference(None, "base64://" + b64(b"definitely-not-an-image"), max_bytes=4096, block_private=True)

    async def test_base64_oversize_raises(self):
        with self.assertRaisesRegex(ReferenceImageError, "过大"):
            await fetch_reference(None, "base64://" + b64(PNG), max_bytes=16, block_private=True)

    async def test_local_path_still_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ref.png"
            path.write_bytes(PNG)
            image = await fetch_reference(None, str(path), max_bytes=4096, block_private=True)
            self.assertEqual(image.mime_type, "image/png")
            self.assertEqual(image.data, PNG)

    async def test_data_url_still_supported(self):
        image = await fetch_reference(None, f"data:image/png;base64,{b64(PNG)}", max_bytes=4096, block_private=True)
        self.assertEqual(image.mime_type, "image/png")
        self.assertEqual(image.data, PNG)

    def test_detect_image_mime_magic_numbers(self):
        self.assertEqual(detect_image_mime(PNG), "image/png")
        self.assertEqual(detect_image_mime(JPEG), "image/jpeg")
        self.assertEqual(detect_image_mime(WEBP), "image/webp")
        self.assertEqual(detect_image_mime(GIF), "image/gif")
        self.assertIsNone(detect_image_mime(b"plain text"))
        self.assertIsNone(detect_image_mime(b""))
        self.assertIsNone(detect_image_mime(b"RIFFxxxxAVI " + bytes(8)))

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

    async def test_local_path_image_is_taken_over_without_converter(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ref.png"
            path.write_bytes(PNG)
            local, deferred = await self.planner().plan([Image(path=str(path))])
            self.assertEqual(len(local), 1)
            self.assertEqual(local[0].data, PNG)
            self.assertEqual(local[0].mime_type, "image/png")
            self.assertEqual(local[0].filename, path.name)
            self.assertEqual(deferred, [])

    async def test_base64_url_image_is_taken_over_inline_without_converter(self):
        calls = []

        async def converter():
            calls.append("called")
            return ""

        component = Image(url="base64://" + b64(PNG), converter=converter)
        local, deferred = await self.planner().plan([component])
        self.assertEqual(len(local), 1)
        self.assertEqual(local[0].data, PNG)
        self.assertEqual(local[0].mime_type, "image/png")
        self.assertEqual(local[0].filename, "base64-reference.png")
        self.assertEqual(calls, [])
        self.assertEqual(deferred, [])

    async def test_data_url_image_is_taken_over_inline_without_converter(self):
        calls = []

        async def converter():
            calls.append("called")
            return ""

        component = Image(url=f"data:image/png;base64,{b64(PNG)}", converter=converter)
        local, deferred = await self.planner().plan([component])
        self.assertEqual(len(local), 1)
        self.assertEqual(local[0].data, PNG)
        self.assertEqual(local[0].mime_type, "image/png")
        self.assertEqual(calls, [])
        self.assertEqual(deferred, [])

    async def test_remote_url_image_defers_strict_without_converter(self):
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

    async def test_remote_url_in_file_field_also_defers_strict(self):
        component = Image(file="https://example.invalid/a.png")
        local, deferred = await self.planner().plan([component])
        self.assertEqual(local, [])
        self.assertEqual(len(deferred), 1)
        self.assertEqual(deferred[0]["kind"], "source")
        self.assertEqual(deferred[0]["source"], "https://example.invalid/a.png")
        self.assertTrue(deferred[0]["strict"])

    async def test_bare_filename_image_with_converter_is_taken_over(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ref.png"
            path.write_bytes(PNG)

            async def converter():
                return str(path)

            component = Image(file="0d2bb1468a87d64414f8e563cc61c33c.png", converter=converter)
            local, deferred = await self.planner().plan([component])
            self.assertEqual(len(local), 1)
            self.assertEqual(local[0].data, PNG)
            self.assertEqual(deferred, [])

    async def test_unresolvable_image_raises_strictly(self):
        component = Image(file="0d2bb1468a87d64414f8e563cc61c33c.png")
        with self.assertRaises(ValueError):
            await self.planner().plan([component])

    async def test_failed_converter_raises_strictly(self):
        async def broken():
            raise OSError("download failed")

        component = Image(file="0d2bb1468a87d64414f8e563cc61c33c.png", converter=broken)
        with self.assertRaises(ValueError):
            await self.planner().plan([component])

    async def test_invalid_base64_image_raises_strictly(self):
        component = Image(url="base64://!!not-base64!!")
        with self.assertRaises(ReferenceImageError):
            await self.planner().plan([component])

    async def test_oversize_base64_image_raises_strictly(self):
        with self.assertRaises(ReferenceImageError):
            await self.planner(max_upload=16).plan([Image(url="base64://" + b64(PNG))])

    async def test_duplicate_images_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ref.png"
            path.write_bytes(PNG)
            local, _ = await self.planner().plan([
                Image(path=str(path)),
                Image(path=str(path)),
            ])
            self.assertEqual(len(local), 1)

    async def test_duplicate_inline_and_local_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ref.png"
            path.write_bytes(PNG)
            local, _ = await self.planner().plan([
                Image(url="base64://" + b64(PNG)),
                Image(path=str(path)),
            ])
            self.assertEqual(len(local), 1)

    async def test_text_urls_become_non_strict_deferred_sources(self):
        text = "参考 https://example.invalid/a.png 和 https://example.invalid/b.png"
        local, deferred = await self.planner().plan([FakePlain(text=text)])
        self.assertEqual(local, [])
        self.assertEqual([item["kind"] for item in deferred], ["source", "source"])
        self.assertEqual(
            [item["source"] for item in deferred],
            ["https://example.invalid/a.png", "https://example.invalid/b.png"],
        )
        self.assertTrue(all(item["strict"] is False for item in deferred))

    async def test_reply_without_embedded_image_defers_when_extractor_available(self):
        async def extractor(event, component):
            return []

        reply = Reply(message_str="[图片]")
        local, deferred = await self.planner(extractor=extractor).plan([reply])
        self.assertEqual(local, [])
        self.assertEqual(len(deferred), 1)
        self.assertEqual(deferred[0]["kind"], "reply")
        self.assertTrue(deferred[0]["strict"])

    async def test_reply_with_embedded_image_is_walked_and_taken_over(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ref.png"
            path.write_bytes(PNG)
            reply = Reply(chain=[Image(path=str(path))])
            local, deferred = await self.planner().plan([reply])
            self.assertEqual(len(local), 1)
            self.assertEqual(local[0].data, PNG)
            self.assertEqual(deferred, [])

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

    async def test_reply_id_only_with_empty_chain_text_is_strict(self):
        """Fix A：chain 存在但文本为空同样视为纯引用，不能宽松。"""
        async def extractor(event, component):
            return []

        reply = Reply(id="msg_quoted_002", chain=[FakePlain(text="")], message_str="")
        local, deferred = await self.planner(extractor=extractor).plan([reply])
        self.assertEqual(len(deferred), 1)
        self.assertTrue(deferred[0]["strict"])

    async def test_reply_with_chain_text_without_image_is_not_strict(self):
        """Fix A：有可检查正文且不含图片标记时保持宽松（明确不含图）。"""
        async def extractor(event, component):
            return []

        reply = Reply(id="msg_quoted_003", chain=[FakePlain(text="这是文字引用，没有图片")], message_str="")
        local, deferred = await self.planner(extractor=extractor).plan([reply])
        self.assertEqual(len(deferred), 1)
        self.assertEqual(deferred[0]["kind"], "reply")
        self.assertFalse(deferred[0]["strict"])

    async def test_reply_with_chain_placeholder_image_is_strict(self):
        """Fix A：chain 文本出现 [图片] 占位仍按 strict 处理。"""
        async def extractor(event, component):
            return []

        reply = Reply(id="msg_quoted_004", chain=[FakePlain(text="[图片]")], message_str="")
        local, deferred = await self.planner(extractor=extractor).plan([reply])
        self.assertEqual(len(deferred), 1)
        self.assertTrue(deferred[0]["strict"])

    async def test_reply_without_id_and_without_text_stays_permissive(self):
        """无引用 id 时没有可提取来源，保持宽松不阻断（与既有行为一致）。"""
        async def extractor(event, component):
            return []

        reply = Reply(chain=[], message_str="")
        local, deferred = await self.planner(extractor=extractor).plan([reply])
        self.assertEqual(len(deferred), 1)
        self.assertFalse(deferred[0]["strict"])

class OpenAIImageEditsFieldTests(unittest.TestCase):
    def test_edits_multipart_uses_image_bracket_field(self):
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

class FetchReferenceRedirectTests(unittest.IsolatedAsyncioTestCase):
    """HTTP(S) 重定向 SSRF 加固的行为级测试（IP 字面量，无需 DNS/网络）。"""

    async def test_http_without_redirect_returns_image(self):
        session = FakeRedirectSession({
            "http://1.1.1.1/a.png": FakeResponse(status=200, headers={"Content-Type": "image/png"}, chunks=PNG),
        })
        image = await fetch_reference(session, "http://1.1.1.1/a.png", max_bytes=4096, block_private=True)
        self.assertEqual(image.data, PNG)
        self.assertEqual(image.mime_type, "image/png")
        self.assertEqual([url for url, _ in session.calls], ["http://1.1.1.1/a.png"])
        self.assertIs(session.calls[0][1].get("allow_redirects"), False)

    async def test_redirect_is_followed_manually_and_relative_location_resolved(self):
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
        session = FakeRedirectSession({
            "http://1.1.1.1/a.png": FakeResponse(status=302, headers={"Location": "http://127.0.0.1:8080/ref.png"}),
        })
        with self.assertRaisesRegex(ReferenceImageError, "重定向目标不安全"):
            await fetch_reference(session, "http://1.1.1.1/a.png", max_bytes=4096, block_private=True)
        # 内网目标未被请求（校验在请求前拦截）。
        self.assertEqual([url for url, _ in session.calls], ["http://1.1.1.1/a.png"])

    async def test_redirect_to_link_local_metadata_is_blocked(self):
        session = FakeRedirectSession({
            "http://1.1.1.1/a.png": FakeResponse(status=302, headers={"Location": "http://169.254.169.254/latest/meta-data/"}),
        })
        with self.assertRaises(ReferenceImageError):
            await fetch_reference(session, "http://1.1.1.1/a.png", max_bytes=4096, block_private=True)

    async def test_redirect_with_embedded_credentials_is_blocked(self):
        session = FakeRedirectSession({
            "http://1.1.1.1/a.png": FakeResponse(status=302, headers={"Location": "http://user:pass@1.1.1.1/ref.png"}),
        })
        with self.assertRaises(ReferenceImageError):
            await fetch_reference(session, "http://1.1.1.1/a.png", max_bytes=4096, block_private=True)

    async def test_redirect_to_non_http_scheme_is_blocked(self):
        session = FakeRedirectSession({
            "http://1.1.1.1/a.png": FakeResponse(status=302, headers={"Location": "ftp://1.1.1.1/ref.png"}),
        })
        with self.assertRaises(ReferenceImageError):
            await fetch_reference(session, "http://1.1.1.1/a.png", max_bytes=4096, block_private=True)

    async def test_redirect_without_location_is_rejected(self):
        session = FakeRedirectSession({
            "http://1.1.1.1/a.png": FakeResponse(status=302, headers={}),
        })
        with self.assertRaisesRegex(ReferenceImageError, "缺少 Location"):
            await fetch_reference(session, "http://1.1.1.1/a.png", max_bytes=4096, block_private=True)

    async def test_redirect_limit_exceeded_is_rejected(self):
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
        session = FakeRedirectSession({})
        with self.assertRaises(ReferenceImageError):
            await fetch_reference(session, "http://127.0.0.1/ref.png", max_bytes=4096, block_private=True)
        self.assertEqual(session.calls, [])


class FetchReferenceHttpContentTests(unittest.IsolatedAsyncioTestCase):
    """HTTP 参考图内容校验：空 body、魔数识别与 verify_magic 开关。"""

    async def test_http_empty_body_is_rejected(self):
        session = FakeRedirectSession({
            "http://1.1.1.1/empty.png": FakeResponse(status=200, headers={"Content-Type": "image/png"}, chunks=b""),
        })
        with self.assertRaisesRegex(ReferenceImageError, "无法获取"):
            await fetch_reference(session, "http://1.1.1.1/empty.png", max_bytes=4096, block_private=True)

    async def test_http_non_image_magic_is_rejected(self):
        session = FakeRedirectSession({
            "http://1.1.1.1/fake.png": FakeResponse(status=200, headers={"Content-Type": "image/png"}, chunks=b"<html>not an image</html>"),
        })
        with self.assertRaisesRegex(ReferenceImageError, "远程响应不是图片"):
            await fetch_reference(session, "http://1.1.1.1/fake.png", max_bytes=4096, block_private=True)

    async def test_http_magic_overrides_header_mime(self):
        session = FakeRedirectSession({
            "http://1.1.1.1/a.gif": FakeResponse(status=200, headers={"Content-Type": "image/gif"}, chunks=JPEG),
        })
        image = await fetch_reference(session, "http://1.1.1.1/a.gif", max_bytes=4096, block_private=True)
        self.assertEqual(image.mime_type, "image/jpeg")
        self.assertEqual(image.data, JPEG)

    async def test_http_verify_magic_false_trusts_header(self):
        # 生成结果下载路径保持宽松：信任 Content-Type 声明，不校验魔数。
        session = FakeRedirectSession({
            "http://1.1.1.1/a.avif": FakeResponse(status=200, headers={"Content-Type": "image/avif"}, chunks=b"arbitrary-avif-bytes"),
        })
        image = await fetch_reference(session, "http://1.1.1.1/a.avif", max_bytes=4096, block_private=True, verify_magic=False)
        self.assertEqual(image.mime_type, "image/avif")

    async def test_data_url_verify_magic_false_uses_declared_mime(self):
        image = await fetch_reference(None, f"data:image/png;base64,{b64(b'not-an-image')}", max_bytes=4096, block_private=True, verify_magic=False)
        self.assertEqual(image.mime_type, "image/png")

class FetchReferenceDataUrlTests(unittest.IsolatedAsyncioTestCase):
    """data URL 魔数识别与 base64 解码前长度预估。"""

    async def test_data_url_mime_mismatch_uses_magic(self):
        image = await fetch_reference(None, f"data:image/png;base64,{b64(JPEG)}", max_bytes=4096, block_private=True)
        self.assertEqual(image.mime_type, "image/jpeg")

    async def test_data_url_unrecognized_magic_raises(self):
        with self.assertRaisesRegex(ReferenceImageError, "无法识别"):
            await fetch_reference(None, f"data:image/png;base64,{b64(b'not an image')}", max_bytes=4096, block_private=True)

    async def test_data_url_oversize_is_rejected_before_decode(self):
        with self.assertRaisesRegex(ReferenceImageError, "过大"):
            await fetch_reference(None, f"data:image/png;base64,{b64(PNG)}", max_bytes=16, block_private=True)

    async def test_data_url_matches_magic_still_ok(self):
        image = await fetch_reference(None, f"data:image/png;base64,{b64(PNG)}", max_bytes=4096, block_private=True)
        self.assertEqual(image.mime_type, "image/png")
        self.assertEqual(image.data, PNG)

class ResolveCheckedUrlTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_ip_literal_is_pinned_with_host_header(self):
        from imago.core.network import _resolve_checked_url
        request_url, headers = await _resolve_checked_url("http://1.1.1.1/a.png", block_private=True)
        self.assertEqual(request_url, "http://1.1.1.1/a.png")
        self.assertEqual(headers, {"Host": "1.1.1.1"})

    async def test_http_port_is_kept_in_netloc_and_host_header(self):
        from imago.core.network import _resolve_checked_url
        request_url, headers = await _resolve_checked_url("http://1.1.1.1:8080/x", block_private=True)
        self.assertEqual(request_url, "http://1.1.1.1:8080/x")
        self.assertEqual(headers, {"Host": "1.1.1.1:8080"})

    async def test_private_ip_is_rejected(self):
        from imago.core.network import _resolve_checked_url
        with self.assertRaisesRegex(ValueError, "私网"):
            await _resolve_checked_url("http://127.0.0.1/x", block_private=True)

    async def test_private_allowed_when_block_disabled(self):
        from imago.core.network import _resolve_checked_url
        request_url, headers = await _resolve_checked_url("http://127.0.0.1/x", block_private=False)
        self.assertEqual(request_url, "http://127.0.0.1/x")

    async def test_https_keeps_original_url_without_host_header(self):
        from imago.core.network import _resolve_checked_url
        request_url, headers = await _resolve_checked_url("https://1.1.1.1/a.png", block_private=True)
        self.assertEqual(request_url, "https://1.1.1.1/a.png")
        self.assertIsNone(headers)


if __name__ == "__main__":
    unittest.main()
