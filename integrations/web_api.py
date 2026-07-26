from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping

from quart import jsonify, request

from ..core.config import load_config
from ..core.errors import DuplicateImage


def _ok(message: str = "ok", **data):
    return jsonify({"success": True, "message": message, **data})


def _err(message: str, status: int = 400):
    return jsonify({"success": False, "message": message}), status


def _line_items(value) -> list[str]:
    values = value if isinstance(value, list) else str(value or "").splitlines()
    return [str(item).strip() for item in values if str(item).strip()]


class PageAPI:
    def __init__(self, plugin):
        self.plugin = plugin

    async def personas(self):
        try:
            return _ok(personas=[{"id": value} for value in await self.plugin.list_persona_ids()])
        except Exception:
            return _err("无法读取 Persona 列表", 500)

    async def detail(self):
        persona_id = str(request.args.get("persona_id", "")).strip()
        if not persona_id or len(persona_id) > 256:
            return _err("persona_id 无效")
        try:
            prompt = await self.plugin.get_persona_prompt(persona_id)
            if not prompt:
                return _err("Persona 不存在或 prompt 为空", 404)
            item = self.plugin.store.get_summary(persona_id, prompt)
            return _ok(
                persona_id=persona_id,
                prompt=prompt,
                summary=(item or {}).get("summary", ""),
                summary_manual=bool((item or {}).get("manual", False)),
                references=self.plugin.store.list_references(persona_id),
            )
        except Exception:
            return _err("无法读取 Persona 素材", 500)

    async def upload(self):
        try:
            body = await request.get_json(force=True)
            persona_id = str((body or {}).get("persona_id", "")).strip()
            if not await self.plugin.get_persona_prompt(persona_id):
                return _err("Persona 不存在", 404)
            images = (body or {}).get("images", [])
            if not isinstance(images, list) or not images:
                return _err("请选择至少一张图片")
            added = []
            for image in images:
                data = base64.b64decode(str(image.get("data", "")), validate=True)
                added.append(self.plugin.store.add_reference(persona_id, data, str(image.get("mime_type", ""))))
            return _ok("上传成功", references=added)
        except DuplicateImage as exc:
            return _err(str(exc), 409)
        except (ValueError, binascii.Error):
            return _err("上传数据或图片格式无效")
        except Exception:
            return _err("上传失败", 500)

    async def delete(self):
        try:
            body = await request.get_json(force=True)
            self.plugin.store.delete_reference(str((body or {}).get("persona_id", "")), str((body or {}).get("filename", "")))
            return _ok("已删除")
        except (ValueError, FileNotFoundError):
            return _err("文件不存在或请求不安全", 404)
        except Exception:
            return _err("删除失败", 500)

    async def preview(self):
        try:
            path = self.plugin.store.reference_path(str(request.args.get("persona_id", "")), str(request.args.get("filename", "")))
            mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}.get(path.suffix.lower(), "image/png")
            encoded = base64.b64encode(path.read_bytes()).decode()
            return _ok(mime_type=mime, base64_data=encoded, data_url=f"data:{mime};base64,{encoded}")
        except (ValueError, FileNotFoundError):
            return _err("图片不存在", 404)
        except Exception:
            return _err("无法读取图片", 500)

    async def rebuild(self):
        try:
            body = await request.get_json(force=True)
            persona_id = str((body or {}).get("persona_id", "")).strip()
            reference_names = (body or {}).get("reference_names", [])
            if not isinstance(reference_names, list):
                return _err("参考图选择格式无效")
            summary = await self.plugin.rebuild_summary(persona_id, reference_names)
            return _ok("已重建", summary=summary)
        except Exception:
            return _err("无法重建外观摘要；请确认 Persona prompt 和副脑 Chat Provider 已配置", 500)

    async def set_summary(self):
        try:
            body = await request.get_json(force=True)
            persona_id = str((body or {}).get("persona_id", "")).strip()
            summary = str((body or {}).get("summary", "")).strip()
            prompt = await self.plugin.get_persona_prompt(persona_id)
            if not prompt:
                return _err("Persona 不存在", 404)
            item = self.plugin.store.set_summary(persona_id, prompt, summary, manual=True)
            return _ok("手工外观摘要已保存", summary=item["summary"])
        except ValueError as exc:
            return _err(str(exc))
        except Exception:
            return _err("无法保存外观摘要", 500)

    async def providers(self):
        raw_items = self.plugin.raw_config.get("providers", []) or []
        configured = []
        seen = set()
        for index, item in enumerate(raw_items):
            if not isinstance(item, Mapping):
                continue
            provider_id = str(item.get("id", "")).strip()
            if not provider_id:
                provider_id = f"未命名节点 {index + 1}"
            keys = _line_items(item.get("api_keys", []))
            has_keys = bool(keys)
            complete = bool(str(item.get("id", "")).strip() and str(item.get("base_url", "")).strip() and has_keys and provider_id not in seen)
            configured.append({
                "id": provider_id,
                "api_type": str(item.get("api_type", "")),
                "model": str(item.get("model", "")),
                "reference_image_limit": max(0, int(item.get("reference_image_limit", 0) or 0)),
                "available_models": [str(value) for value in (item.get("available_models", []) or [])],
                "base_url": str(item.get("base_url", "")),
                "api_keys_configured": has_keys,
                "default_size": str(item.get("default_size", "1024x1024")),
                "timeout": max(10, int(item.get("timeout", 180) or 180)),
                "complete": complete,
            })
            seen.add(provider_id)
        primary = self.plugin.store.get_primary_provider_id()
        raw_ids = [item["id"] for item in configured]
        effective = primary if primary in raw_ids else (raw_ids[0] if raw_ids else "")
        chat_providers = []
        for provider in self.plugin.context.get_all_providers():
            try:
                meta = provider.meta()
                chat_providers.append({"id": str(meta.id), "model": str(meta.model or ""), "type": str(meta.type)})
            except Exception:
                continue
        optimizer = self.plugin.raw_config.get("optimizer_config", {}) or {}
        return _ok(
            providers=configured,
            primary_provider_id=effective,
            stored_primary_provider_id=primary,
            chat_providers=chat_providers,
            optimizer_provider_id=str(optimizer.get("optimizer_provider_id", "")),
            vision_provider_id=str(optimizer.get("vision_provider_id", "")),
        )

    async def save_providers(self):
        try:
            body = await request.get_json(force=True)
            incoming = (body or {}).get("providers", [])
            primary = str((body or {}).get("primary_provider_id", "")).strip()
            optimizer_provider_id = str((body or {}).get("optimizer_provider_id", "")).strip()
            vision_provider_id = str((body or {}).get("vision_provider_id", "")).strip()
            if not isinstance(incoming, list):
                return _err("节点配置格式无效")
            old_items = self.plugin.raw_config.get("providers", []) or []
            old_by_id = {str(item.get("id", "")).strip(): item for item in old_items if isinstance(item, Mapping)}
            saved = []
            seen = set()
            for item in incoming:
                if not isinstance(item, Mapping):
                    return _err("节点配置格式无效")
                provider_id = str(item.get("id", "")).strip()
                if not provider_id or provider_id in seen:
                    return _err("节点 ID 不能为空或重复")
                api_type = str(item.get("api_type", "")).strip()
                if api_type not in {"openai_image", "openai_chat", "gemini_official", "dashscope_multimodal", "custom_endpoint"}:
                    return _err(f"节点 {provider_id} 的接口类型无效")
                old = old_by_id.get(provider_id, {})
                submitted_keys = _line_items(item.get("api_keys", []))
                stored_keys = _line_items(old.get("api_keys", []))
                saved.append({
                    "id": provider_id,
                    "api_type": api_type,
                    "base_url": str(item.get("base_url", "")).strip(),
                    "api_keys": submitted_keys if submitted_keys else stored_keys,
                    "model": str(item.get("model", "")).strip(),
                    "reference_image_limit": max(0, int(item.get("reference_image_limit", 0) or 0)),
                    "available_models": [str(value).strip() for value in (item.get("available_models", []) or []) if str(value).strip()],
                    "default_size": str(item.get("default_size", "1024x1024")).strip() or "1024x1024",
                    "timeout": max(10, int(item.get("timeout", 180) or 180)),
                })
                seen.add(provider_id)
            if primary and primary not in seen:
                return _err("主节点不在当前列表中")
            self.plugin.raw_config["providers"] = saved
            optimizer = self.plugin.raw_config.get("optimizer_config", {}) or {}
            optimizer["optimizer_provider_id"] = optimizer_provider_id
            optimizer["vision_provider_id"] = vision_provider_id
            self.plugin.raw_config["optimizer_config"] = optimizer
            save_fn = getattr(self.plugin.raw_config, "save_config", None)
            if not callable(save_fn):
                return _err("当前 AstrBot 配置对象不支持持久化", 500)
            save_fn()
            if primary:
                self.plugin.store.set_primary_provider_id(primary)
            return _ok("节点配置已保存", primary_provider_id=primary)
        except (TypeError, ValueError):
            return _err("超时和参考图上限必须是有效整数")
        except Exception:
            return _err("无法保存节点配置", 500)

    async def quotas(self):
        try:
            policy = load_config(self.plugin.raw_config).quota
            return _ok(
                quotas=self.plugin.quota_store.list_rows(policy),
                quota_enabled=policy.enabled,
                daily_refresh_enabled=policy.daily_refresh_enabled,
                daily_quota_target=policy.daily_quota_target,
                checkin_enabled=policy.checkin_enabled,
                checkin_quota_min=policy.checkin_quota_min,
                checkin_quota_max=policy.checkin_quota_max,
            )
        except Exception:
            return _err("无法读取绘图额度", 500)

    async def save_quotas(self):
        try:
            body = await request.get_json(force=True)
            items = (body or {}).get("quotas", [])
            if not isinstance(items, list) or len(items) > 5000:
                return _err("额度列表格式无效")
            rows = self.plugin.quota_store.set_many(
                items,
                load_config(self.plugin.raw_config).quota,
            )
            return _ok("绘图额度已保存", quotas=rows)
        except (TypeError, ValueError) as exc:
            return _err(str(exc))
        except Exception:
            return _err("无法保存绘图额度", 500)

    async def set_primary_provider(self):
        try:
            body = await request.get_json(force=True)
            provider_id = str((body or {}).get("provider_id", "")).strip()
            configured = [str(item.get("id", "")).strip() for item in (self.plugin.raw_config.get("providers", []) or []) if isinstance(item, Mapping)]
            if provider_id not in configured:
                return _err("图片节点不存在", 404)
            self.plugin.store.set_primary_provider_id(provider_id)
            return _ok("主图片节点已更新", primary_provider_id=provider_id)
        except Exception:
            return _err("无法保存主图片节点", 500)
