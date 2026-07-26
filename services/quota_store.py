from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import re
import threading
import time
from typing import Callable, Iterable

from ..core.models import QuotaConfig, TaskState


USER_ID = re.compile(r"^[A-Za-z0-9_.:@-]{1,128}$")


def terminal_refund_amount(state: TaskState, charged: int) -> int:
    """只有普通 failed 退回本次实际扣除额度。"""
    return max(0, int(charged)) if state == TaskState.FAILED else 0


@dataclass(frozen=True, slots=True)
class QuotaSnapshot:
    user_id: str
    quota: int
    blocked: bool
    unlimited: bool
    quota_enabled: bool
    last_refresh_date: str = ""
    last_checkin_date: str = ""


@dataclass(frozen=True, slots=True)
class QuotaDecision:
    allowed: bool
    snapshot: QuotaSnapshot
    charged: int = 0
    reason: str = ""


@dataclass(frozen=True, slots=True)
class CheckinResult:
    success: bool
    snapshot: QuotaSnapshot
    reward: int = 0
    reason: str = ""


class QuotaStore:
    def __init__(
        self,
        root: Path,
        today_provider: Callable[[], str],
        randint: Callable[[int, int], int] = random.randint,
    ):
        self.path = root / "quotas.json"
        self.today_provider = today_provider
        self.randint = randint
        self._lock = threading.RLock()

    @staticmethod
    def normalize_user_id(user_id: str) -> str:
        value = str(user_id or "").strip()
        if not USER_ID.fullmatch(value):
            raise ValueError("用户 ID 无效")
        return value

    @staticmethod
    def _quota(value) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    def _load(self) -> dict:
        try:
            value = json.loads(self.path.read_text("utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            value = {}
        users = value.get("users", {}) if isinstance(value, dict) else {}
        return {
            "version": 1,
            "users": users if isinstance(users, dict) else {},
        }

    def _save(self, value: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), "utf-8")
        os.replace(temporary, self.path)

    def _record(self, users: dict, user_id: str) -> dict:
        raw = users.get(user_id)
        record = raw if isinstance(raw, dict) else {}
        record = {
            "quota": self._quota(record.get("quota", 0)),
            "last_refresh_date": str(record.get("last_refresh_date", "")),
            "last_checkin_date": str(record.get("last_checkin_date", "")),
            "updated_at": self._quota(record.get("updated_at", 0)),
        }
        users[user_id] = record
        return record

    def _refresh(self, record: dict, policy: QuotaConfig, today: str) -> bool:
        if not policy.daily_refresh_enabled or record["last_refresh_date"] == today:
            return False
        # 每日刷新是“重置到目标值”：低于目标时补足，高于目标时削减。
        record["quota"] = policy.daily_quota_target
        record["last_refresh_date"] = today
        record["updated_at"] = int(time.time())
        return True

    @staticmethod
    def _snapshot(user_id: str, record: dict, policy: QuotaConfig) -> QuotaSnapshot:
        blocked = user_id in policy.blacklist_ids
        unlimited = not blocked and user_id in policy.unlimited_whitelist_ids
        return QuotaSnapshot(
            user_id=user_id,
            quota=int(record.get("quota", 0)),
            blocked=blocked,
            unlimited=unlimited,
            quota_enabled=policy.enabled,
            last_refresh_date=str(record.get("last_refresh_date", "")),
            last_checkin_date=str(record.get("last_checkin_date", "")),
        )

    @classmethod
    def _policy_snapshot(cls, user_id: str, policy: QuotaConfig) -> QuotaSnapshot:
        """Build an access-only snapshot without creating a persisted user row."""
        return cls._snapshot(user_id, {}, policy)

    def inspect(self, user_id: str, policy: QuotaConfig) -> QuotaSnapshot:
        user_id = self.normalize_user_id(user_id)
        with self._lock:
            value = self._load()
            record = self._record(value["users"], user_id)
            if self._refresh(record, policy, self.today_provider()):
                self._save(value)
            return self._snapshot(user_id, record, policy)

    def can_consume(self, user_id: str, amount: int, policy: QuotaConfig) -> QuotaDecision:
        user_id = self.normalize_user_id(user_id)
        amount = max(1, int(amount))
        policy_snapshot = self._policy_snapshot(user_id, policy)
        if policy_snapshot.blocked:
            return QuotaDecision(False, policy_snapshot, reason="你当前无法使用绘图功能。")
        if policy_snapshot.unlimited or not policy.enabled:
            return QuotaDecision(True, policy_snapshot)
        snapshot = self.inspect(user_id, policy)
        if snapshot.quota < amount:
            return QuotaDecision(False, snapshot, reason=f"绘图额度不足：需要 {amount}，当前剩余 {snapshot.quota}。")
        return QuotaDecision(True, snapshot)

    def consume(self, user_id: str, amount: int, policy: QuotaConfig) -> QuotaDecision:
        user_id = self.normalize_user_id(user_id)
        amount = max(1, int(amount))
        policy_snapshot = self._policy_snapshot(user_id, policy)
        if policy_snapshot.blocked:
            return QuotaDecision(False, policy_snapshot, reason="你当前无法使用绘图功能。")
        if policy_snapshot.unlimited or not policy.enabled:
            return QuotaDecision(True, policy_snapshot)
        with self._lock:
            value = self._load()
            record = self._record(value["users"], user_id)
            self._refresh(record, policy, self.today_provider())
            snapshot = self._snapshot(user_id, record, policy)
            if snapshot.quota < amount:
                self._save(value)
                return QuotaDecision(False, snapshot, reason=f"绘图额度不足：需要 {amount}，当前剩余 {snapshot.quota}。")
            record["quota"] -= amount
            record["updated_at"] = int(time.time())
            self._save(value)
            return QuotaDecision(True, self._snapshot(user_id, record, policy), charged=amount)

    def adjust(self, user_id: str, operation: str, amount: int, policy: QuotaConfig) -> QuotaSnapshot:
        user_id = self.normalize_user_id(user_id)
        amount = int(amount)
        if amount < 0:
            raise ValueError("额度不能小于 0")
        if operation not in {"add", "del", "set"}:
            raise ValueError("额度操作无效")
        with self._lock:
            value = self._load()
            record = self._record(value["users"], user_id)
            today = self.today_provider()
            self._refresh(record, policy, today)
            if operation == "add":
                record["quota"] += amount
            elif operation == "del":
                record["quota"] = max(0, record["quota"] - amount)
            else:
                record["quota"] = amount
            record["last_refresh_date"] = today
            record["updated_at"] = int(time.time())
            self._save(value)
            return self._snapshot(user_id, record, policy)

    def refund(self, user_id: str, amount: int, policy: QuotaConfig) -> QuotaSnapshot:
        """退回一次任务实际扣除的额度，不受当前黑白名单或启用开关影响。"""
        user_id = self.normalize_user_id(user_id)
        amount = max(0, int(amount))
        with self._lock:
            value = self._load()
            record = self._record(value["users"], user_id)
            today = self.today_provider()
            self._refresh(record, policy, today)
            if amount:
                record["quota"] += amount
                record["updated_at"] = int(time.time())
            record["last_refresh_date"] = today
            self._save(value)
            return self._snapshot(user_id, record, policy)

    def checkin(self, user_id: str, policy: QuotaConfig) -> CheckinResult:
        user_id = self.normalize_user_id(user_id)
        with self._lock:
            value = self._load()
            record = self._record(value["users"], user_id)
            today = self.today_provider()
            self._refresh(record, policy, today)
            snapshot = self._snapshot(user_id, record, policy)
            if snapshot.blocked:
                self._save(value)
                return CheckinResult(False, snapshot, reason="你当前无法使用绘图功能。")
            if not policy.checkin_enabled:
                self._save(value)
                return CheckinResult(False, snapshot, reason="签到额度未启用。")
            if snapshot.unlimited:
                self._save(value)
                return CheckinResult(False, snapshot, reason="你已在无限额度白名单中，无需签到。")
            if record["last_checkin_date"] == today:
                self._save(value)
                return CheckinResult(False, snapshot, reason="今天已经签到过了。")
            reward = self.randint(policy.checkin_quota_min, policy.checkin_quota_max)
            record["quota"] += reward
            record["last_checkin_date"] = today
            record["updated_at"] = int(time.time())
            self._save(value)
            return CheckinResult(True, self._snapshot(user_id, record, policy), reward=reward)

    def list_rows(self, policy: QuotaConfig) -> list[dict]:
        with self._lock:
            value = self._load()
            users = value["users"]
            ids = set(users) | set(policy.blacklist_ids) | set(policy.unlimited_whitelist_ids)
            changed = False
            today = self.today_provider()
            rows = []
            for user_id in sorted(ids):
                try:
                    user_id = self.normalize_user_id(user_id)
                except ValueError:
                    continue
                record = self._record(users, user_id)
                changed = self._refresh(record, policy, today) or changed
                snapshot = self._snapshot(user_id, record, policy)
                rows.append({
                    "user_id": snapshot.user_id,
                    "quota": snapshot.quota,
                    "blocked": snapshot.blocked,
                    "unlimited": snapshot.unlimited,
                    "last_refresh_date": snapshot.last_refresh_date,
                    "last_checkin_date": snapshot.last_checkin_date,
                })
            if changed:
                self._save(value)
            return rows

    def set_many(self, items: Iterable[dict], policy: QuotaConfig) -> list[dict]:
        normalized = []
        seen = set()
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("额度数据格式无效")
            user_id = self.normalize_user_id(item.get("user_id", ""))
            if user_id in seen:
                raise ValueError("用户 ID 重复")
            seen.add(user_id)
            quota = int(item.get("quota", 0))
            if quota < 0:
                raise ValueError("额度不能小于 0")
            normalized.append((user_id, quota))
        with self._lock:
            value = self._load()
            today = self.today_provider()
            for user_id, quota in normalized:
                record = self._record(value["users"], user_id)
                record["quota"] = quota
                record["last_refresh_date"] = today
                record["updated_at"] = int(time.time())
            self._save(value)
        return self.list_rows(policy)
