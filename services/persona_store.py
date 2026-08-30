from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from ..core.errors import DuplicateImage
from ..core.security import ensure_child, safe_component

ALLOWED_MIME = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}


class PersonaStore:
    def __init__(
        self,
        root: Path,
        max_upload_bytes: int | Callable[[], int] = 20 * 1024 * 1024,
    ):
        self.root = root
        self.summaries = root / "summaries.json"
        self.settings = root / "settings.json"
        self.references = root / "persona_references"
        self.task_cache = root / "task_cache"
        self._max_upload_bytes = max_upload_bytes
        self.references.mkdir(parents=True, exist_ok=True)
        self.task_cache.mkdir(parents=True, exist_ok=True)

    @property
    def max_upload_bytes(self) -> int:
        value = self._max_upload_bytes() if callable(self._max_upload_bytes) else self._max_upload_bytes
        return max(1, int(value))

    @staticmethod
    def _task_id(task_id: str) -> str:
        value = str(task_id).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{16,64}", value):
            raise ValueError("任务 ID 不安全")
        return value

    def task_dir(self, task_id: str) -> Path:
        return ensure_child(self.task_cache, self.task_cache / self._task_id(task_id))

    def task_output_dir(self, task_id: str) -> Path:
        path = self.task_dir(task_id) / "outputs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def update_task_manifest(self, task_id: str, **fields) -> dict:
        directory = self.task_dir(task_id)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "task.json"
        try:
            value = json.loads(target.read_text("utf-8"))
            value = value if isinstance(value, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            value = {}
        value.setdefault("task_id", self._task_id(task_id))
        value.setdefault("created_at", int(time.time()))
        value.update(fields)
        value["updated_at"] = int(time.time())
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), "utf-8")
        os.replace(temporary, target)
        return value

    def persist_task_inputs(self, task_id: str, images) -> list[dict]:
        directory = self.task_dir(task_id) / "inputs"
        directory.mkdir(parents=True, exist_ok=True)
        records = []
        suffixes = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}
        for index, image in enumerate(images, start=1):
            digest = hashlib.sha256(image.data).hexdigest()
            suffix = suffixes.get(image.mime_type, ".bin")
            filename = f"{index:02d}_{digest[:16]}{suffix}"
            target = ensure_child(directory, directory / filename)
            target.write_bytes(image.data)
            records.append({"file": filename, "mime_type": image.mime_type, "size": len(image.data), "sha256": digest})
        self.update_task_manifest(task_id, inputs=records, input_count=len(records), input_bytes=sum(item["size"] for item in records))
        return records

    def record_task_outputs(self, task_id: str, paths: list[Path]) -> list[dict]:
        records = []
        for path in paths:
            if not path.is_file() or path.is_symlink():
                continue
            records.append({"file": path.name, "size": path.stat().st_size})
        self.update_task_manifest(task_id, outputs=records, output_count=len(records), output_bytes=sum(item["size"] for item in records))
        return records

    def prune_task_cache(self, max_bytes: int, protected: set[str] | None = None) -> None:
        limit = max(16 * 1024 * 1024, int(max_bytes))
        protected = {self._task_id(value) for value in (protected or set())}
        directories = [path for path in self.task_cache.iterdir() if path.is_dir() and path.name not in protected]
        total = sum(path.stat().st_size for path in self.task_cache.rglob("*") if path.is_file())
        if total <= limit:
            return
        for directory in sorted(directories, key=lambda path: path.stat().st_mtime):
            size = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
            shutil.rmtree(directory, ignore_errors=True)
            total -= size
            if total <= limit:
                break

    @staticmethod
    def prompt_hash(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _load_summaries(self) -> dict[str, dict]:
        try:
            value = json.loads(self.summaries.read_text("utf-8"))
            return value if isinstance(value, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _save_summaries(self, value: dict[str, dict]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.summaries.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), "utf-8")
            os.replace(temporary, self.summaries)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def get_primary_provider_id(self) -> str:
        try:
            value = json.loads(self.settings.read_text("utf-8"))
            return str(value.get("primary_provider_id", "")).strip() if isinstance(value, dict) else ""
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return ""

    def set_primary_provider_id(self, provider_id: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.settings.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps({"primary_provider_id": provider_id}, ensure_ascii=False, indent=2), "utf-8")
            os.replace(temporary, self.settings)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def get_summary(self, persona_id: str, prompt: str) -> dict | None:
        item = self._load_summaries().get(persona_id)
        if not item:
            return None
        if not item.get("manual") and item.get("source_hash") != self.prompt_hash(prompt):
            return None
        if not item.get("manual") and item.get("uses_references"):
            names = item.get("reference_names", []) or []
            if item.get("reference_hash", "") != self.reference_fingerprint(persona_id, names):
                return None
        return item

    def set_summary(self, persona_id: str, prompt: str, summary: str, *, manual: bool, reference_names=None) -> dict:
        summary = " ".join(summary.split())
        if not manual:
            summary = summary[:220]
        if not summary:
            raise ValueError("外观摘要不能为空")
        reference_names = [str(value) for value in (reference_names or [])]
        values = self._load_summaries()
        values[persona_id] = item = {
            "summary": summary,
            "source_hash": self.prompt_hash(prompt),
            "reference_hash": self.reference_fingerprint(persona_id, reference_names),
            "reference_names": reference_names,
            "uses_references": bool(reference_names),
            "updated_at": int(time.time()),
            "manual": bool(manual),
        }
        self._save_summaries(values)
        return item

    def reference_fingerprint(self, persona_id: str, names=None) -> str:
        names = sorted(str(value) for value in (names if names is not None else [item["name"] for item in self.list_references(persona_id)]))
        if names:
            for name in names:
                try:
                    self.reference_path(persona_id, name)
                except (ValueError, FileNotFoundError):
                    return "missing"
        return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest() if names else ""

    def persona_dir(self, persona_id: str) -> Path:
        return self.references / safe_component(persona_id)

    def add_reference(self, persona_id: str, data: bytes, mime_type: str) -> dict[str, str | int]:
        if mime_type not in ALLOWED_MIME or not data or len(data) > self.max_upload_bytes:
            raise ValueError("图片格式或大小不符合要求")
        digest = hashlib.md5(data, usedforsecurity=False).hexdigest()
        directory = self.persona_dir(persona_id)
        directory.mkdir(parents=True, exist_ok=True)
        if any(path.name.startswith(digest + ".") for path in directory.iterdir()):
            raise DuplicateImage("该 Persona 已有相同内容的图片")
        target = directory / f"{digest}{ALLOWED_MIME[mime_type]}"
        target.write_bytes(data)
        return {"name": target.name, "md5": digest, "size": len(data), "mime_type": mime_type}

    def list_references(self, persona_id: str) -> list[dict[str, int | str]]:
        directory = self.persona_dir(persona_id)
        if not directory.exists():
            return []
        return [{"name": p.name, "size": p.stat().st_size} for p in sorted(directory.iterdir()) if p.is_file() and not p.is_symlink()]

    def reference_path(self, persona_id: str, filename: str) -> Path:
        if Path(filename).name != filename:
            raise ValueError("文件名不安全")
        directory = self.persona_dir(persona_id)
        path = ensure_child(directory, directory / filename)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(filename)
        return path

    def delete_reference(self, persona_id: str, filename: str) -> None:
        self.reference_path(persona_id, filename).unlink()
