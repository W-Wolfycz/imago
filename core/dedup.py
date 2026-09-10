"""单次请求内的重复调用防护（纯逻辑，便于单测）。

主 LLM 在同一个 agent 轮次里可能对同一画面连续调用两次绘图工具（工具返回后
没有按指令停止），造成重复生成与重复扣费。这里以「触发消息 ID」为主键判定：
同一条用户消息只允许创建一个绘图任务，措辞被改写也一律拦掉；用户新发一条
消息（ID 不同）则允许重新生成。触发消息 ID 不可用时退化为「同内容 + 时间窗」。
"""

from __future__ import annotations

import hashlib

# 触发消息 ID 不可用时的退化时间窗（秒）：同内容任务结束后仍在该窗口内视为重复。
# 与 scheduler 的终态保留期一致（在途任务不受窗口限制，见 records 的 pending）。
DEDUPE_WINDOW_SECONDS = 30.0


def request_fingerprint(*parts) -> str:
    """按内容生成请求指纹：仅用于缺少触发消息 ID 时的退化判重与日志诊断。"""
    normalized = "\n".join(" ".join(str(part if part is not None else "").split()) for part in parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def is_duplicate_request(
    records,
    *,
    dedupe_key: str,
    trigger_message_id: str = "",
    now: float,
    window: float = DEDUPE_WINDOW_SECONDS,
) -> bool:
    """是否已有同一条触发消息创建的任务（或缺少消息 ID 时的同内容任务）。

    records 每项含 dedupe_key / trigger_message_id / created_at，可带 pending
    （任务是否仍在处理中）：
    - 双方都有触发消息 ID：只比较 ID，相同即判重复（与措辞无关）；
    - 任一侧缺 ID：退化为「同内容指纹 +（仍在处理中，或结束时间在退化窗口内）」。
      在途任务不受窗口长度限制——出图可能远超窗口，仍在跑就是重复。
    """
    for record in records or []:
        record_message_id = str(record.get("trigger_message_id", ""))
        if trigger_message_id and record_message_id:
            if record_message_id == trigger_message_id:
                return True
            continue
        if str(record.get("dedupe_key", "")) != dedupe_key:
            continue
        if record.get("pending"):
            return True
        try:
            created = float(record.get("created_at") or 0.0)
        except (TypeError, ValueError):
            created = 0.0
        if now - created <= window:
            return True
    return False
