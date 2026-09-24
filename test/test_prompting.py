import unittest

from imago.core.prompting import (
    CAMERA_REQUEST_MARKER,
    BASIS_LABELS,
    STYLE_LLM_CHOICES,
    STYLE_AUTO_CATALOG,
    DEFAULT_CAMERA_SUFFIX,
    REFERENCE_RELATION_SUFFIX,
    STYLE_GUIDANCE,
    STYLE_PROMPT_SUFFIX,
    compose_persona_prompt,
    merge_camera_request,
    optimizer_system,
    reference_relation_suffix,
    resolve_style,
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
    def test_fallback_block_never_injects_optimizer_meta_prompt(self):
        # 副脑元指令（optimizer_prompt）不得进入图片 prompt：降级块只含风格
        # 预设与默认视角后缀。
        meta = "将本轮画面需求整理为准确、具体、可直接交给图片模型的提示词"
        prompt = compose_persona_prompt("s", "d", style="realistic", fallback_suffix=True)
        self.assertNotIn(meta, prompt)
        self.assertNotIn("低优先级自定义风格", prompt)
        self.assertIn(DEFAULT_CAMERA_SUFFIX, prompt)
        self.assertIn(STYLE_PROMPT_SUFFIX["realistic"], prompt)


class ReferenceRelationSuffixTests(unittest.TestCase):
    def test_suffix_distinguishes_roles(self):
        self.assertEqual(reference_relation_suffix(2, 0), REFERENCE_RELATION_SUFFIX)
        suffix = reference_relation_suffix(1, 3)
        self.assertIn(REFERENCE_RELATION_SUFFIX, suffix)
        self.assertIn("first 1 attached image(s)", suffix)
        self.assertIn("remaining 3 image(s)", suffix)
        self.assertIn("character identity references", suffix)



class AutoStyleTableTests(unittest.TestCase):
    """选择表的基准项必须与基准表一一对应：少一项会让某个基准在 auto 下永远选不出来。"""

    def test_table_keys_match_basis_labels(self):
        self.assertEqual([key for key, _, _ in STYLE_AUTO_CATALOG], list(BASIS_LABELS))
        self.assertNotIn("default", [key for key, _, _ in STYLE_AUTO_CATALOG])


class ResolveStyleTests(unittest.TestCase):
    """风格联动只发生在代码写死的 auto 选项上。"""

    def test_configured_preset_always_wins(self):
        for requested in ("", "auto", "pixel", "none", "bogus"):
            with self.subTest(requested=requested):
                self.assertEqual(resolve_style("realistic", requested), "realistic")

    def test_auto_links_to_llm_preset(self):
        # 主 LLM 只能传表中的基准之一
        self.assertEqual(STYLE_LLM_CHOICES, tuple(BASIS_LABELS))
        self.assertEqual(resolve_style("auto", "pixel"), "pixel")
        self.assertEqual(resolve_style("auto", " PIXEL "), "pixel")
        self.assertEqual(resolve_style("auto", "logo"), "logo")
        # 空/auto/default/none/非法值都落回 auto（交副脑按用户措辞选）
        for requested in ("", "auto", "default", "none", "steampunk", "anime style"):
            with self.subTest(requested=requested):
                self.assertEqual(resolve_style("auto", requested), "auto")


class StyleAuthorityTests(unittest.TestCase):
    """基准由插件裁决后交给副脑执行：具体基准锁定，auto 交副脑选，none/default 不锁。"""

    def test_bases_are_locked_and_excluded_from_user_priority(self):
        for style, name in BASIS_LABELS.items():
            with self.subTest(style=style):
                self.assertTrue(style_is_locked(style))
                prompt = optimizer_system("", style, persona=True)
                self.assertIn(f"本轮成像基准已固定为「{name}」", prompt)
                self.assertIn(f"「{name}」", optimizer_system("", style, persona=False))
                # 口径文本确实进了 prompt（漏注入属静默失效）
                self.assertIn(STYLE_GUIDANCE[style], prompt)
                # 基准已锁定：用户优先条款里不再给"媒介"，避免与基准块互相打架
                self.assertNotIn("指定的媒介", prompt)

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
                self.assertIn("HARD EXCLUSIONS", STYLE_GUIDANCE[style])
                for marker in markers:
                    self.assertIn(marker, STYLE_GUIDANCE[style])

    def test_real3d_basis_claims_humans_and_collectible_toys(self):
        # real3d 原口径只写"物"：写实三维人物/数字人会被 auto 判去真人实拍，手办潮玩被当通用效果
        # 塞进别的基准。这是协议条款，删掉不报错但 auto 行为静默回退。
        basis = STYLE_GUIDANCE["real3d"]
        # 断言主体清单里的独家措辞：排除项里也有 "designer toy"，只断言单词会被漏改蒙过去
        for phrase in ("digital human", "designer toys and ornaments"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, basis)
        # 降级后缀是另一条注入路径（副脑没跑时生效），同样要认人物主体
        self.assertIn("digital human", STYLE_PROMPT_SUFFIX["real3d"])


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
