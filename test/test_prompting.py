import unittest

from imago.core.prompting import (
    provider_supports_image,
    usable_summary,
    BASIS_LABELS,
    STYLE_GUIDANCE,
    compose_persona_prompt,
    summary_user_prompt,
    vision_user_prompt,
)


class SummaryUsabilityTests(unittest.TestCase):
    def test_rejection_reply_is_not_a_summary(self):
        # 拒答被当摘要存下＝角色此后一直用这句话出图，页面还标成「自动摘要」，很难查
        self.assertFalse(usable_summary("没有可用的外观资料，无法提取稳定外观特征。"))
        self.assertFalse(usable_summary("EMPTY"))
        self.assertTrue(usable_summary("白发少女，长直发扎成双马尾，深红瞳孔，白色露肩连衣裙。"))
        # 反问有两种措辞（您/你），靠句式清单必然漏，问号才是稳定信号
        self.assertFalse(usable_summary(
            "我已经收到这张角色参考图的外观证据，但您还没有给出具体任务。您希望我接下来做什么？"
        ))
        self.assertFalse(usable_summary(
            "已收到参考图的外观与氛围信息。请告诉我你希望我基于这些信息产出什么，例如 1. 图像生成提示词"
        ))

    def test_provider_supports_image_reads_modalities(self):
        # 识图 Provider 指到纯文本模型＝模型看不到图，只会回一句"暂无视觉资料"
        self.assertFalse(provider_supports_image({"modalities": ["text"]}))
        self.assertTrue(provider_supports_image({"modalities": ["text", "image"]}))
        # 没声明模态时不拦，避免把可用配置误判成不支持
        self.assertIsNone(provider_supports_image({}))

class SummaryPromptShapeTests(unittest.TestCase):
    def test_summary_prompt_states_output_shape_and_optional_persona(self):
        # 格式靠提示词约束：没说清输出形状或能否缺人设，模型就自由发挥成列表/反问
        from imago.core.prompting import SUMMARY_SYSTEM
        self.assertIn("单段落", SUMMARY_SYSTEM)
        self.assertIn("可能不存在", SUMMARY_SYSTEM)

    def test_empty_inputs_are_omitted_not_left_blank(self):
        # 留空 XML 块会让模型以为"素材给了、任务没给"，于是反问要做什么
        text = summary_user_prompt("", "白发少女，长直发扎成双马尾")
        self.assertNotIn("<persona_prompt>", text)
        self.assertIn("<visual_evidence>", text)

    def test_single_image_prompt_does_not_say_compare(self):
        # 只有一张图时说"比较这 N 张"是错话
        self.assertNotIn("比较", vision_user_prompt(1))
        self.assertIn("这 3 张", vision_user_prompt(3))

    def test_output_rules_are_also_in_user_prompt(self):
        # 形状约束在 user 里也要有一份，不单独依赖 system 通道
        self.assertIn("单段落", vision_user_prompt(1))
        self.assertIn("单段落", summary_user_prompt("人设", "证据"))

class MultiPersonaPromptTests(unittest.TestCase):
    def test_compose_lists_every_character(self):
        # 少列一个角色＝该角色只靠参考图出现，容易变成路人或干脆消失。
        prompt = compose_persona_prompt(
            "主角色摘要", "两人并肩看海", primary_label="演示角色甲",
            others=(("演示角色乙", "白发红瞳"),),
        )
        self.assertIn("- 演示角色甲: 主角色摘要", prompt)
        self.assertIn("- 演示角色乙: 白发红瞳", prompt)
        self.assertIn("Current scene: 两人并肩看海", prompt)




class BasisDifferentiationTests(unittest.TestCase):
    """基准必须强区分：每条口径都写死硬性排除项，且互相点名排除对方的特征。"""

    def test_bases_exclude_each_other_by_name(self):
        # 排除项被删＝两个基准画风互相渗透且无报错。
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


if __name__ == "__main__":
    unittest.main()
