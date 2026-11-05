import unittest

from imago.core.prompting import BASIS_LABELS, STYLE_GUIDANCE






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
