import unittest

from imago.core.prompting import (
    CAMERA_REQUEST_MARKER,
    DEFAULT_CAMERA_SUFFIX,
    REFERENCE_RELATION_SUFFIX,
    STYLE_PROMPT_SUFFIX,
    caption_system_text,
    compose_persona_prompt,
    merge_camera_request,
    optimizer_system,
    reference_relation_suffix,
    sanitize_caption,
)

class CameraTests(unittest.TestCase):
    def test_merge_camera_request(self):
        self.assertEqual(merge_camera_request("海边回头", ""), "海边回头")
        self.assertEqual(merge_camera_request("", ""), "")
        merged = merge_camera_request("海边回头", "自拍")
        self.assertEqual(merged, f"海边回头\n{CAMERA_REQUEST_MARKER}: 自拍")
        self.assertEqual(merge_camera_request("", "俯拍 45 度"), f"{CAMERA_REQUEST_MARKER}: 俯拍 45 度")

class ComposePersonaPromptTests(unittest.TestCase):
    def test_no_fallback_suffix_is_plain_concat(self):
        # 副脑正常完成/开关关闭：纯拼接，不注入任何后缀。
        for prompt in (
            compose_persona_prompt("s", "d"),
            compose_persona_prompt("s", "d", style=""),
            compose_persona_prompt("s", "d", style="realistic"),
        ):
            self.assertEqual(prompt, "Character identity (stable): s\nCurrent scene: d")

    def test_fallback_block_never_injects_optimizer_meta_prompt(self):
        # 副脑元指令（optimizer_prompt）不得进入图片 prompt：降级块只含风格
        # 预设与默认视角后缀。
        meta = "将本轮画面需求整理为准确、具体、可直接交给图片模型的提示词"
        prompt = compose_persona_prompt("s", "d", style="realistic", fallback_suffix=True)
        self.assertNotIn(meta, prompt)
        self.assertNotIn("低优先级自定义风格", prompt)
        self.assertIn(DEFAULT_CAMERA_SUFFIX, prompt)
        self.assertIn(STYLE_PROMPT_SUFFIX["realistic"], prompt)


class PersonaOptimizerProtocolTests(unittest.TestCase):
    def test_persona_protocol_defaults_to_third_person_view(self):
        prompt = optimizer_system("", "default", persona=True)
        self.assertIn("第三方视角", prompt)
        self.assertIn("他拍", prompt)
        self.assertIn("避免默认怼脸自拍或特写", prompt)
        self.assertIn("以用户为准", prompt)

class CaptionSanitizeTests(unittest.TestCase):
    def test_sanitize_caption_collapses_whitespace_and_truncates(self):
        self.assertEqual(sanitize_caption("  画好了\n\n看看喜欢吗  "), "画好了 看看喜欢吗")
        self.assertEqual(sanitize_caption("长文" * 100, max_length=10), "长文" * 5)

class ReferenceRelationSuffixTests(unittest.TestCase):
    def test_suffix_distinguishes_roles(self):
        self.assertEqual(reference_relation_suffix(2, 0), REFERENCE_RELATION_SUFFIX)
        suffix = reference_relation_suffix(1, 3)
        self.assertIn(REFERENCE_RELATION_SUFFIX, suffix)
        self.assertIn("first 1 attached image(s)", suffix)
        self.assertIn("remaining 3 image(s)", suffix)
        self.assertIn("character identity references", suffix)


class CaptionSystemTextTests(unittest.TestCase):
    def test_no_images_forbids_success_tone(self):
        text = caption_system_text("人设A", has_images=False)
        self.assertIn("没有生成任何图片", text)
        self.assertIn("不要声称图片已准备好", text)
        self.assertNotIn("图片拼接在文字末尾", text)
        with_images = caption_system_text("人设A", has_images=True)
        self.assertIn("图片拼接在文字末尾", with_images)

if __name__ == "__main__":
    unittest.main()
