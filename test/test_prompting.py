import unittest

from imago.core.prompting import (
    CAMERA_REQUEST_MARKER,
    DEFAULT_CAMERA_SUFFIX,
    STYLE_GUIDANCE,
    STYLE_PROMPT_SUFFIX,
    compose_persona_prompt,
    merge_camera_request,
    optimizer_system,
    persona_prompt_suffix,
    sanitize_caption,
    style_prompt_suffix,
)


class StyleGuidanceTests(unittest.TestCase):
    def test_realistic_reinforced_photography_terms(self):
        guidance = STYLE_GUIDANCE["realistic"].lower()
        for keyword in (
            "micro-texture",
            "pores",
            "beauty-filter",
            "smoothing",
            "glass/plastic skin",
            "lens focal length",
            "exposure",
            "film grain",
            "candid posing",
        ):
            with self.subTest(keyword=keyword):
                self.assertIn(keyword, guidance)
        # 转译后仍保留可辨识身份
        self.assertIn("recognizable", guidance)
        self.assertIn("face shape", guidance)
        self.assertIn("hair color", guidance)

    def test_anime_preserves_identity(self):
        guidance = STYLE_GUIDANCE["anime"].lower()
        self.assertIn("recognizable", guidance)
        self.assertIn("face shape", guidance)
        self.assertIn("hair color", guidance)
        self.assertIn("generic faces", guidance)

    def test_3d_preserves_likeness_and_avoids_uncanny_valley(self):
        guidance = STYLE_GUIDANCE["3d"].lower()
        self.assertIn("likeness", guidance)
        self.assertIn("recognizable", guidance)
        self.assertIn("uncanny valley", guidance)
        self.assertIn("generic game faces", guidance)

    def test_cinematic_mentions_35mm_full_frame(self):
        guidance = STYLE_GUIDANCE["cinematic"].lower()
        self.assertIn("35mm", guidance)
        self.assertIn("full-frame", guidance)


class StyleSuffixTests(unittest.TestCase):
    def test_style_prompt_suffix_only_for_non_default_styles(self):
        for style in ("realistic", "cinematic", "anime", "3d"):
            with self.subTest(style=style):
                self.assertTrue(style_prompt_suffix(style))
        for style in ("none", "default", "", "unknown"):
            with self.subTest(style=style):
                self.assertEqual(style_prompt_suffix(style), "")

    def test_suffix_is_low_priority_and_defers_to_user(self):
        for style, suffix in STYLE_PROMPT_SUFFIX.items():
            with self.subTest(style=style):
                self.assertIn("low-priority", suffix.lower())
                self.assertIn("does not conflict with the user", suffix.lower())
                self.assertIn("explicit", suffix.lower())


class CameraTests(unittest.TestCase):
    def test_default_camera_suffix_is_third_person_and_user_priority(self):
        text = DEFAULT_CAMERA_SUFFIX.lower()
        self.assertIn("third-person", text)
        self.assertIn("not in the frame", text)
        self.assertIn("selfie", text)
        self.assertIn("close-up", text)
        self.assertIn("follow the user", text)
        self.assertIn("explicitly specified", text)

    def test_merge_camera_request_empty_returns_action_unchanged(self):
        self.assertEqual(merge_camera_request("海边回头", ""), "海边回头")
        self.assertEqual(merge_camera_request("海边回头", "  "), "海边回头")
        self.assertEqual(merge_camera_request("", ""), "")

    def test_merge_camera_request_appends_marked_request(self):
        merged = merge_camera_request("海边回头", "自拍")
        self.assertEqual(merged, f"海边回头\n{CAMERA_REQUEST_MARKER}: 自拍")
        self.assertEqual(merge_camera_request("", "俯拍 45 度"), f"{CAMERA_REQUEST_MARKER}: 俯拍 45 度")

    def test_persona_prompt_suffix_combines_style_and_default_camera(self):
        block = persona_prompt_suffix("realistic")
        self.assertIn(DEFAULT_CAMERA_SUFFIX, block)
        self.assertIn(STYLE_PROMPT_SUFFIX["realistic"], block)
        # none/default 只保留默认视角后缀
        self.assertEqual(persona_prompt_suffix("none"), DEFAULT_CAMERA_SUFFIX)
        self.assertEqual(persona_prompt_suffix("default"), DEFAULT_CAMERA_SUFFIX)


class ComposePersonaPromptTests(unittest.TestCase):
    def test_style_suffix_falls_into_final_prompt_on_fallback(self):
        prompt = compose_persona_prompt("stable summary", "scene", style="realistic", fallback_suffix=True)
        self.assertIn("Character identity (stable): stable summary", prompt)
        self.assertIn("Current scene: scene", prompt)
        self.assertIn(DEFAULT_CAMERA_SUFFIX, prompt)
        self.assertIn(STYLE_PROMPT_SUFFIX["realistic"], prompt)

    def test_default_style_fallback_appends_only_camera_suffix(self):
        for style in ("none", "default"):
            with self.subTest(style=style):
                prompt = compose_persona_prompt("s", "d", style=style, fallback_suffix=True)
                self.assertIn(DEFAULT_CAMERA_SUFFIX, prompt)
                for suffix in STYLE_PROMPT_SUFFIX.values():
                    self.assertNotIn(suffix, prompt)

    def test_no_fallback_suffix_is_plain_concat(self):
        # 副脑正常完成/开关关闭：纯拼接，不注入任何后缀。
        for prompt in (
            compose_persona_prompt("s", "d"),
            compose_persona_prompt("s", "d", style=""),
            compose_persona_prompt("s", "d", style="realistic"),
        ):
            self.assertEqual(prompt, "Character identity (stable): s\nCurrent scene: d")

    def test_custom_prompt_falls_into_fallback_block(self):
        prompt = compose_persona_prompt("s", "d", custom_prompt="低优先级自定义风格", fallback_suffix=True)
        self.assertIn("低优先级自定义风格", prompt)
        self.assertIn(DEFAULT_CAMERA_SUFFIX, prompt)


class PersonaOptimizerProtocolTests(unittest.TestCase):
    def test_persona_protocol_defaults_to_third_person_view(self):
        prompt = optimizer_system("", "default", persona=True)
        self.assertIn("第三方视角", prompt)
        self.assertIn("他拍", prompt)
        self.assertIn("避免默认怼脸自拍或特写", prompt)
        self.assertIn("以用户为准", prompt)

    def test_persona_protocol_preserves_existing_contract(self):
        prompt = optimizer_system("CUSTOM", "none", persona=True)
        self.assertIn("CUSTOM", prompt)
        self.assertIn("<identity_summary>", prompt)
        self.assertIn("不得复制", prompt)


class CaptionSanitizeTests(unittest.TestCase):
    def test_sanitize_caption_collapses_whitespace_and_truncates(self):
        self.assertEqual(sanitize_caption("  画好了\n\n看看喜欢吗  "), "画好了 看看喜欢吗")
        self.assertEqual(sanitize_caption("长文" * 100, max_length=10), "长文" * 5)

    def test_sanitize_caption_keeps_content_as_is(self):
        # 内容清洗交给发送装饰链（ChatMemory 等），imago 不做替换。
        self.assertEqual(sanitize_caption("看，[图片]！"), "看，[图片]！")

    def test_sanitize_caption_empty(self):
        self.assertEqual(sanitize_caption("  "), "")


if __name__ == "__main__":
    unittest.main()
