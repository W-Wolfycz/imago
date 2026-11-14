from __future__ import annotations

import difflib
import re

# 归一化时丢掉的分隔符与标点：人设名常见「示例系列——演示角色甲」「(Laffey)」「·」这类写法。
_SEPARATORS = re.compile(
    r"[\s\u3000\-—–‑_·・|/\\()（）\[\]【】<>《》\"'`,，。.:：;；!！?？*+&~]+"
)
# 拆分「系列——角色」：只按空白与连接符切，保留标点原样以便还原。
_SERIES_SPLIT = re.compile(r"[\s\u3000\-—–_·・|/]+")


def normalize_name(value: str) -> str:
    """归一化人设名：去掉空白与标点、统一小写，仅用于匹配比较。"""
    return _SEPARATORS.sub("", str(value or "")).lower()


def match_personas(query: str, candidates, *, limit: int = 8) -> list[str]:
    """在候选人设名里找 ``query``，返回原文名（保持调用方拿到的写法）。

    匹配顺序：完全相同 → 去掉系列前缀后相同（`示例系列——演示角色乙` 对 `演示角色乙`）→ 双向子串。
    只做本地字符串匹配，不做拼音/编辑距离：填错字就返回空，让上层明确报错，
    避免猜错角色后静默画出另一个人。
    """
    target = normalize_name(query)
    if not target:
        return []
    names = [str(item) for item in candidates if str(item or "").strip()]
    exact = [name for name in names if normalize_name(name) == target]
    if exact:
        return exact[:limit]
    segmented = [
        name for name in names
        if any(normalize_name(part) == target for part in _SERIES_SPLIT.split(name))
    ]
    if segmented:
        return segmented[:limit]
    partial = [
        name for name in names
        if target in normalize_name(name) or normalize_name(name) in target
    ]
    return partial[:limit]


def allocate_reference_budget(budget: int, sizes, *, per_group_min: int = 1) -> list[int]:
    """把参考图名额按角色轮转分配，每个角色先拿 ``per_group_min`` 张再轮流补。

    名额不足以覆盖全部角色时按传入顺序给前面的角色（调用方把主角色排在首位）；
    返回长度与 ``sizes`` 相同，总和不超过 ``budget``。
    """
    sizes = [max(0, int(size)) for size in sizes]
    remaining = max(0, int(budget))
    quotas = [0] * len(sizes)
    for index, size in enumerate(sizes):
        if remaining <= 0:
            break
        take = min(per_group_min, size, remaining)
        quotas[index] = take
        remaining -= take
    while remaining > 0:
        progressed = False
        for index, size in enumerate(sizes):
            if remaining <= 0:
                break
            if quotas[index] < size:
                quotas[index] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return quotas


def suggest_personas(
    query: str, candidates, *, limit: int = 3, cutoff: float = 0.34,
) -> list[str]:
    """按字面相似度列出最接近的候选，只用于「你是不是想找这个」的提示。

    不参与匹配决策：祥瑞 / 甲鹤 / 乙鹤 是三个不同角色，把打错的名字自动替换成
    另一个名字等于静默画错人，要等出图才发现。所以这里只给线索，由调用方决定。
    """
    target = normalize_name(query)
    if not target:
        return []
    scored = []
    for name in candidates:
        text = normalize_name(name)
        if not text:
            continue
        ratio = difflib.SequenceMatcher(None, target, text).ratio()
        # 共享汉字是弱信号：乙鹤 / 甲鹤 共享「鹤」，比纯字面比例稳一点。
        shared = len(set(target) & set(text))
        # 同位置相同的字更强：乙鹤 vs 甲鹤 末字都在同一个位置，
        # 而祥瑞只是把「瑞」挪到了开头，不该排得更前。
        same_position = sum(1 for left, right in zip(target, text) if left == right)
        if ratio >= cutoff or shared >= max(1, len(target) // 2):
            scored.append((ratio + shared * 0.1 + same_position * 0.2, name))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, name in scored[:limit]]


def companion_capacity(max_companions: int, reference_slots: int) -> int:
    """可同框的客串角色数：受全局上限与节点参考图名额共同约束。

    ``reference_slots`` 为节点一次能发几张参考图，0 表示不限；每个角色至少要占 1 张，
    否则它会分到 0 张、只靠文字出镜（出图后才发现"少了一个人"）。
    """
    limit = max(0, int(max_companions))
    slots = max(0, int(reference_slots or 0))
    if slots <= 0:
        return limit
    return min(limit, max(0, slots - 1))
