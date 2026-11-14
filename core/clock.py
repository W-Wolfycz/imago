"""时区日期键（纯逻辑，便于单测）。"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


def day_key(timezone_name: str, now: datetime | None = None) -> tuple[str, bool]:
    """按给定时区取 ISO 日期键，返回 ``(日期, 时区是否解析失败)``。

    时区为空或非法时回落到本地时区，并把失败标记交给调用方——静默回落会让日切
    错位（额度刷新与每日签到按错误的"今天"计算，差几小时就可能多领或漏领）。
    """
    text = str(timezone_name or "").strip()
    if text:
        try:
            zone = ZoneInfo(text)
        except Exception:
            return (now or datetime.now()).date().isoformat(), True
        current = now or datetime.now(zone)
        if current.tzinfo is None:
            current = current.replace(tzinfo=zone)
        return current.astimezone(zone).date().isoformat(), False
    return (now or datetime.now()).date().isoformat(), False
