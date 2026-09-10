import unittest

from imago.core.prompting import (
    CAMERA_REQUEST_MARKER,
    AUTO_NEUTRAL_ENTRY,
    BASIS_LABELS,
    FIXED_STYLE_PRESETS,
    STYLE_LLM_CHOICES,
    STYLE_AUTO_CATALOG,
    DEFAULT_CAMERA_SUFFIX,
    REFERENCE_RELATION_SUFFIX,
    STYLE_GUIDANCE,
    STYLE_PROMPT_SUFFIX,
    caption_system_text,
    compose_persona_prompt,
    merge_camera_request,
    optimizer_system,
    reference_relation_suffix,
    resolve_style,
    sanitize_caption,
    style_is_locked,
    style_prompt_suffix,
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


class AutoStyleSelectionTests(unittest.TestCase):
    """auto 的表：只有成像基准 + 一个"不指定基准"的通用兜底项。"""

    def test_auto_table_lists_every_basis_plus_neutral_fallback(self):
        prompt = optimizer_system("", "auto", persona=True)
        self.assertIn("基准选择", prompt)
        for label in ("REALISTIC_PHOTO", "REAL_3D", "CG_3D", "HAND_DRAWN_ILLUSTRATION",
                      "PIXEL_GRID", "LOGO_DESIGN"):
            with self.subTest(label=label):
                self.assertIn(label, prompt)
        # 表的基准项恰好等于 BASIS_LABELS，且不含通用兜底（兜底只由 AUTO_NEUTRAL_ENTRY 提供）
        self.assertEqual([key for key, _, _ in STYLE_AUTO_CATALOG], list(BASIS_LABELS))
        self.assertNotIn("default", [key for key, _, _ in STYLE_AUTO_CATALOG])
        # 无线索 → 通用（不指定基准），不硬塞某个基准
        self.assertIn("选表末的 GENERAL_NEUTRAL", prompt)
        self.assertIn(AUTO_NEUTRAL_ENTRY, prompt)
        # 不写死条目数量：加基准时提示词自动跟随
        self.assertNotIn("四选一", prompt)
        # 单一基准模式不出现选择表
        self.assertNotIn("基准选择", optimizer_system("", "pixel", persona=True))


class ResolveStyleTests(unittest.TestCase):
    """风格联动只发生在代码写死的 auto 选项上。"""

    def test_configured_preset_always_wins(self):
        for requested in ("", "auto", "pixel", "none", "bogus"):
            with self.subTest(requested=requested):
                self.assertEqual(resolve_style("realistic", requested), "realistic")

    def test_auto_links_to_llm_preset(self):
        # 主 LLM 只能传六个基准之一
        self.assertEqual(STYLE_LLM_CHOICES, tuple(BASIS_LABELS))
        self.assertEqual(resolve_style("auto", "pixel"), "pixel")
        self.assertEqual(resolve_style("auto", " PIXEL "), "pixel")
        self.assertEqual(resolve_style("auto", "logo"), "logo")
        # 空/auto/default/none/非法值都落回 auto（交副脑按用户措辞选）
        for requested in ("", "auto", "default", "none", "steampunk", "anime style"):
            with self.subTest(requested=requested):
                self.assertEqual(resolve_style("auto", requested), "auto")

    def test_blank_config_falls_back_to_default(self):
        self.assertEqual(resolve_style("", ""), "default")
        self.assertEqual(resolve_style("", "pixel"), "default")


class StyleAuthorityTests(unittest.TestCase):
    """基准由插件裁决后交给副脑执行：具体基准锁定，auto 交副脑选，none/default 不锁。"""

    def test_bases_are_locked_and_excluded_from_user_priority(self):
        for style, name in BASIS_LABELS.items():
            with self.subTest(style=style):
                self.assertTrue(style_is_locked(style))
                prompt = optimizer_system("", style, persona=True)
                self.assertIn(f"本轮成像基准已固定为「{name}」", prompt)
                self.assertIn(STYLE_GUIDANCE[style], prompt)
                # 基准已锁定：不再出现"用户明确指定的媒介优先"，避免与基准块互相打架
                self.assertNotIn("指定的媒介", prompt)
                self.assertIn("用户本轮明确指定的视角和构图", prompt)
                # 学派/题材/光影/效果仍要在基准之上按用户措辞补写（示例在共享句里）
                self.assertIn("学派、题材、光影氛围与效果", prompt)
                self.assertIn("日系赛璐璐", prompt)

    def test_unlocked_options_keep_user_style_priority(self):
        # none 不注入基准块；default 只给通用口径；auto 交副脑按用户措辞选——
        # 三者都不锁定基准，"用户优先"条款里保留风格与媒介。
        def priority_line(style):
            prompt = optimizer_system("", style, persona=False)
            return next(ln for ln in prompt.splitlines() if "始终优先于" in ln)

        for style in ("none", "default", "auto"):
            with self.subTest(style=style):
                self.assertFalse(style_is_locked(style))
                line = priority_line(style)
                self.assertIn("风格", line)
                self.assertIn("媒介", line)
                self.assertNotIn("基准已由插件固定", line)
        self.assertNotIn("风格预设：", optimizer_system("", "none", persona=False))
        self.assertIn("风格预设：", optimizer_system("", "default", persona=False))


class BasisDifferentiationTests(unittest.TestCase):
    """基准必须强区分：每条口径都写死硬性排除项，且互相点名排除对方的特征。"""

    def test_every_style_block_declares_hard_exclusions(self):
        for key, guidance in STYLE_GUIDANCE.items():
            if key == "default":
                continue
            with self.subTest(style=key):
                self.assertIn("HARD EXCLUSIONS", guidance)

    def test_bases_exclude_each_other_by_name(self):
        cases = {
            "realistic": ("cel shading", "pixel grid", "sculpted figurine"),
            "real3d": ("hand-drawn linework", "cartoon or stylized proportions", "pixel grid"),
            "cg3d": ("hand-drawn linework", "painterly brush texture", "visible pixel grid"),
            "illustration": ("photographic optics", "pixel grid", "CGI sheen"),
            "pixel": ("anti-aliased or blurred edges", "photographic detail", "painterly brushwork"),
            "logo": ("photographic realism", "3D bevels", "hand-drawn brush or ink texture"),
        }
        self.assertEqual(set(cases), set(BASIS_LABELS))
        for style, markers in cases.items():
            with self.subTest(style=style):
                for marker in markers:
                    self.assertIn(marker, STYLE_GUIDANCE[style])

    def test_basis_names_are_written_into_the_prompt(self):
        # 副脑要"从基准开始补写"，所以基准必须有明确名字，两条协议里都要有
        for style, name in BASIS_LABELS.items():
            for persona in (True, False):
                with self.subTest(style=style, persona=persona):
                    self.assertIn(f"「{name}」", optimizer_system("", style, persona=persona))


class StylePromptSuffixTests(unittest.TestCase):
    def test_suffix_keys_match_basis_labels_exactly(self):
        # 加了基准却忘记补降级后缀时会被这里拦住；多余的后缀键同样算错
        self.assertEqual(set(STYLE_PROMPT_SUFFIX), set(BASIS_LABELS))
        for style in BASIS_LABELS:
            with self.subTest(style=style):
                self.assertTrue(style_prompt_suffix(style))
        for style in ("", "default", "none", "auto"):
            with self.subTest(style=style):
                self.assertEqual(style_prompt_suffix(style), "")


if __name__ == "__main__":
    unittest.main()
