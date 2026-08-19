"""指令参数类型兼容层（纯逻辑，不引入顶层 astrbot 依赖破坏单元测试）。

AstrBot 的 CommandFilter 会按空格截断普通 ``str`` 参数，只取第一个词；
把最后一个参数注解为 ``GreedyStr`` 后，CommandFilter 会把剩余所有片段用
空格拼接后整体传入。真实 AstrBot 环境使用官方 ``GreedyStr``；在没有
AstrBot 运行依赖的环境（纯逻辑测试）下回退到行为等价的 ``str`` 子类，
保证模块仍可导入。
"""

from __future__ import annotations

try:
    from astrbot.core.star.filter.command import GreedyStr
except ImportError:  # 无 AstrBot 运行依赖时的安全兼容占位，行为等价于 str
    class GreedyStr(str):
        """AstrBot GreedyStr 的本地兼容占位（仅缺 AstrBot 运行时生效）。"""
