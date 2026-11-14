from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping

from quart import jsonify, request

from astrbot.api import logger

from ..core.config import (
    API_TYPES,
    TYPED_KEYS_BY_API,
    load_config,
    provider_custom_params,
    split_custom_params,
)
from ..core.images import thumbnail
from ..core.params import PARAM_FIELDS
from ..core.errors import DuplicateImage


def _ok(message: str = "ok", **data):
    return jsonify({"success": True, "message": message, **data})


def _err(message: str, status: int = 400):
    return jsonify({"success": False, "message": message}), status


# 单次上传 Persona 参考图的数量上限（每张仍有 max_upload_bytes 大小限制）。
MAX_UPLOAD_IMAGES = 20


def _line_items(value) -> list[str]:
    values = value if isinstance(value, list) else str(value or "").splitlines()
    return [str(item).strip() for item in values if str(item).strip()]


def _safe_int(value, default: int, minimum: int | None = None) -> int:
    """配置值转整数并兜底：手工改坏（非数字/null）时回退默认，不抛 500。"""
    try:
        result = int(value or default)
    except (TypeError, ValueError):
        result = default
    if minimum is not None:
        result = max(minimum, result)
    return result


class PageAPI:
    def __init__(self, plugin):
        self.plugin = plugin

    async def personas(self):
        try:
            astrbot_ids = set(await self.plugin.list_persona_ids())
            local_ids = set(self.plugin.store.list_local_names())
            merged = sorted(astrbot_ids | local_ids)
            return _ok(
                personas=[{"id": value} for value in merged],
                sources={
                    value: ("persona" if value in astrbot_ids else "local") for value in merged
                },
            )
        except Exception as exc:
            logger.warning(
                "[Imago] PageAPI %s 失败 type=%s", "personas", type(exc).__name__,
            )
            return _err("无法读取 Persona 列表", 500)

    async def create_local_persona(self):
        """手动新增本地角色：只建名字，不需要人设 Prompt。"""
        try:
            body = await request.get_json(force=True)
            name = str((body or {}).get("name", "")).strip()
            if not name:
                return _err("角色名不能为空")
            result = self.plugin.store.create_local_persona(name)
            return _ok(**result)
        except ValueError as exc:
            return _err(str(exc))
        except Exception as exc:
            logger.warning(
                "[Imago] PageAPI %s 失败 type=%s", "create_local_persona", type(exc).__name__,
            )
            return _err("无法新增本地角色", 500)

    async def delete_local_persona(self):
        """删除本地角色目录；AstrBot 里有人设时拒绝，避免误删与素材无关的东西。"""
        try:
            body = await request.get_json(force=True)
            name = str((body or {}).get("persona_id", "")).strip()
            if not name:
                return _err("persona_id 无效")
            if await self.plugin.get_persona_prompt(name):
                return _err("该角色在 AstrBot 里有人设，不能在这里删除")
            self.plugin.store.delete_local_persona(name)
            return _ok(persona_id=name)
        except FileNotFoundError:
            return _err("本地没有这个角色", 404)
        except Exception as exc:
            logger.warning(
                "[Imago] PageAPI %s 失败 type=%s", "delete_local_persona", type(exc).__name__,
            )
            return _err("无法删除本地角色", 500)

    async def _character_exists(self, persona_id: str) -> bool:
        """角色是否存在：AstrBot 有人设，或本地已有对应目录（只读，不建目录）。

        本地角色没有人设 prompt，用它替代「prompt 非空」的存在性判断，否则上传参考图、
        保存手工摘要这些基础操作都会对本地角色返回「Persona 不存在」。
        """
        if not persona_id:
            return False
        if await self.plugin.get_persona_prompt(persona_id):
            return True
        return persona_id in self.plugin.store.list_local_names()

    async def detail(self):
        persona_id = str(request.args.get("persona_id", "")).strip()
        if not persona_id or len(persona_id) > 256:
            return _err("persona_id 无效")
        try:
            prompt = await self.plugin.get_persona_prompt(persona_id)
            item = self.plugin.store.get_summary(persona_id, prompt)
            references = self.plugin.store.list_references(persona_id)
            # 刚新建的本地角色可能既没图也没摘要，只要目录在就算存在——否则前端拿到 404
            # 会保留上一个角色的内容（实测踩过）。
            if not prompt and not references and not await self._character_exists(persona_id):
                return _err("角色不存在", 404)
            return _ok(
                persona_id=persona_id,
                prompt=prompt,
                source="persona" if prompt else "local",
                summary=(item or {}).get("summary", ""),
                summary_manual=bool((item or {}).get("manual", False)),
                references=references,
            )
        except Exception as exc:
            logger.warning(
                "[Imago] PageAPI %s 失败 type=%s", "detail", type(exc).__name__,
            )
            return _err("无法读取 Persona 素材", 500)

    async def upload(self):
        try:
            body = await request.get_json(force=True)
            persona_id = str((body or {}).get("persona_id", "")).strip()
            if not await self._character_exists(persona_id):
                return _err("角色不存在", 404)
            images = (body or {}).get("images", [])
            if not isinstance(images, list) or not images:
                return _err("请选择至少一张图片")
            if len(images) > MAX_UPLOAD_IMAGES:
                return _err(f"单次最多上传 {MAX_UPLOAD_IMAGES} 张图片")
            added = []
            for image in images:
                data = base64.b64decode(str(image.get("data", "")), validate=True)
                added.append(self.plugin.store.add_reference(persona_id, data, str(image.get("mime_type", ""))))
            return _ok("上传成功", references=added)
        except DuplicateImage as exc:
            return _err(str(exc), 409)
        except (ValueError, binascii.Error):
            return _err("上传数据或图片格式无效")
        except Exception as exc:
            logger.warning(
                "[Imago] PageAPI %s 失败 type=%s", "upload", type(exc).__name__,
            )
            return _err("上传失败", 500)

    async def delete(self):
        try:
            body = await request.get_json(force=True)
            self.plugin.store.delete_reference(str((body or {}).get("persona_id", "")), str((body or {}).get("filename", "")))
            return _ok("已删除")
        except (ValueError, FileNotFoundError):
            return _err("文件不存在或请求不安全", 404)
        except Exception as exc:
            logger.warning(
                "[Imago] PageAPI %s 失败 type=%s", "delete", type(exc).__name__,
            )
            return _err("删除失败", 500)

    async def preview(self):
        try:
            path = self.plugin.store.reference_path(str(request.args.get("persona_id", "")), str(request.args.get("filename", "")))
            mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}.get(path.suffix.lower(), "image/png")
            encoded = base64.b64encode(path.read_bytes()).decode()
            return _ok(mime_type=mime, base64_data=encoded, data_url=f"data:{mime};base64,{encoded}")
        except (ValueError, FileNotFoundError):
            return _err("图片不存在", 404)
        except Exception as exc:
            logger.warning(
                "[Imago] PageAPI %s 失败 type=%s", "preview", type(exc).__name__,
            )
            return _err("无法读取图片", 500)

    async def rebuild(self):
        try:
            body = await request.get_json(force=True)
            persona_id = str((body or {}).get("persona_id", "")).strip()
            reference_names = (body or {}).get("reference_names", [])
            if not persona_id or len(persona_id) > 256:
                return _err("persona_id 无效")
            if not isinstance(reference_names, list):
                return _err("参考图选择格式无效")
            # 页面上刚选的 Provider 直接生效，不必先点「保存配置」再回来重建
            summary = await self.plugin.rebuild_summary(
                persona_id,
                reference_names,
                vision_provider_id=str((body or {}).get("vision_provider_id", "")).strip(),
                optimizer_provider_id=str((body or {}).get("optimizer_provider_id", "")).strip(),
            )
            return _ok("已重建", summary=summary)
        except ValueError as exc:
            return _err(str(exc))
        except Exception as exc:
            logger.warning(
                "[Imago] PageAPI %s 失败 type=%s", "rebuild", type(exc).__name__,
            )
            return _err("无法重建外观摘要；请确认 Persona Prompt 和副脑 Provider 已配置", 500)

    async def set_summary(self):
        try:
            body = await request.get_json(force=True)
            persona_id = str((body or {}).get("persona_id", "")).strip()
            summary = str((body or {}).get("summary", "")).strip()
            if not summary:
                return _err("外观摘要不能为空")
            if not await self._character_exists(persona_id):
                return _err("角色不存在", 404)
            prompt = await self.plugin.get_persona_prompt(persona_id)
            item = self.plugin.store.set_summary(persona_id, prompt, summary, manual=True)
            return _ok("手工外观摘要已保存", summary=item["summary"])
        except ValueError as exc:
            return _err(str(exc))
        except Exception as exc:
            logger.warning(
                "[Imago] PageAPI %s 失败 type=%s", "set_summary", type(exc).__name__,
            )
            return _err("无法保存外观摘要", 500)

    async def task_image(self):
        """任务结果图：``size=thumb`` 给缩略图，``size=full`` 给原图。"""
        try:
            task_id = str(request.args.get("task_id", "")).strip()
            name = str(request.args.get("name", "")).strip()
            mode = str(request.args.get("size", "thumb")).strip().lower()
            path = self.plugin.store.task_output_file(task_id, name)
            data = path.read_bytes()
            mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".webp": "image/webp", ".gif": "image/gif"}.get(path.suffix.lower(), "image/png")
            if mode != "full":
                data, mime = thumbnail(data, mime)
            encoded = base64.b64encode(data).decode()
            return _ok(mime_type=mime, base64_data=encoded, data_url=f"data:{mime};base64,{encoded}")
        except (ValueError, FileNotFoundError):
            return _err("任务图片不存在", 404)
        except Exception as exc:
            logger.warning(
                "[Imago] PageAPI %s 失败 type=%s", "task_image", type(exc).__name__,
            )
            return _err("无法读取任务图片", 500)

    async def tasks(self):
        """最近任务清单（只读 task.json：不含图片内容与身份字段）。"""
        try:
            try:
                limit = max(1, min(200, int(request.args.get("limit", 30))))
            except (TypeError, ValueError):
                limit = 30
            return _ok(tasks=self.plugin.store.list_tasks(limit))
        except Exception as exc:
            logger.warning(
                "[Imago] PageAPI %s 失败 type=%s", "tasks", type(exc).__name__,
            )
            return _err("无法读取任务列表", 500)

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
                "reference_image_limit": _safe_int(item.get("reference_image_limit"), 0, 0),
                "enabled": bool(item.get("enabled", True)),
                "available_models": [str(value) for value in (item.get("available_models", []) or [])],
                "base_url": str(item.get("base_url", "")),
                "api_keys_configured": has_keys,
                "custom_params": "\n".join(
                    f"{key}={value}" for key, value in provider_custom_params(item)),
                "timeout": _safe_int(item.get("timeout"), 180, 10),
                "complete": complete,
            })
            seen.add(provider_id)
        primary = self.plugin.store.get_primary_provider_id()
        active_ids = [item["id"] for item in configured if item["enabled"]]
        effective = primary if primary in active_ids else (active_ids[0] if active_ids else "")
        chat_providers = []
        for provider in self.plugin.context.get_all_providers():
            try:
                meta = provider.meta()
                chat_providers.append({"id": str(meta.id), "model": str(meta.model or ""), "type": str(meta.type)})
            except Exception as exc:
                logger.warning(
                    "[Imago] PageAPI %s 失败 type=%s", "providers", type(exc).__name__,
                )
                continue
        optimizer = self.plugin.raw_config.get("optimizer_config", {}) or {}
        return _ok(
            providers=configured,
            param_fields=PARAM_FIELDS,
            primary_provider_id=effective,
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
                if api_type not in API_TYPES:
                    return _err(f"节点 {provider_id} 的接口类型无效")
                # 页面可以改节点 ID：优先按提交里的 original_id 找回旧记录，
                # 否则改名会把已存密钥当成"新节点"而写成空数组（唯一节点直接停摆）。
                original_id = str(item.get("original_id", "")).strip()
                old = old_by_id.get(original_id) if original_id else None
                if old is None:
                    old = old_by_id.get(provider_id, {})
                submitted_keys = _line_items(item.get("api_keys", []))
                stored_keys = _line_items(old.get("api_keys", []))
                # 旧节点只有 default_size：没写自定义参数时把旧尺寸迁成 size=…，
                # 避免用管理页（或旧版页面）保存一次就把默认尺寸丢掉。
                custom_params = str(item.get("custom_params", "")).strip()
                legacy_size = str(old.get("default_size", "") or "").strip()
                if not custom_params and legacy_size:
                    custom_params = f"size={legacy_size}"
                # 控件字段单独落盘，其余键留在文本里（两种页面共用一份配置）
                param_fields, extra_params = split_custom_params(
                    custom_params, TYPED_KEYS_BY_API.get(api_type, ()))
                saved.append({
                    "id": provider_id,
                    # 页面可能不发这个字段：保留旧值，避免保存一次就把停用节点悄悄启用
                    "enabled": bool(item.get("enabled", old.get("enabled", True))),
                    "api_type": api_type,
                    "base_url": str(item.get("base_url", "")).strip(),
                    "api_keys": submitted_keys if submitted_keys else stored_keys,
                    "model": str(item.get("model", "")).strip(),
                    "reference_image_limit": max(0, int(item.get("reference_image_limit", 0) or 0)),
                    "available_models": [str(value).strip() for value in (item.get("available_models", []) or []) if str(value).strip()],
                    "size": param_fields.get("size", ""),
                    "quality_openai": param_fields.get("quality", "")
                    if api_type == "openai_image" else "",
                    "quality_xai": param_fields.get("quality", "")
                    if api_type == "xai_image" else "",
                    "nsfw": param_fields.get("nsfw", "").lower() in {"true", "1", "yes"}
                    if api_type == "xai_image" else False,
                    "custom_params": extra_params,
                    "timeout": max(10, int(item.get("timeout", 180) or 180)),
                    # AstrBot 管理页校验 template_list 要求条目带模板键；
                    # WebUI 保存同样带上，保证两种 UI 数据互相兼容。
                    "__template_key": "provider",
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
        except Exception as exc:
            logger.warning(
                "[Imago] PageAPI %s 失败 type=%s", "save_providers", type(exc).__name__,
            )
            return _err("无法保存节点配置", 500)

    async def save_brain_providers(self):
        """只保存副脑 / 识图 Provider：人设页换了下拉立即生效，不牵动节点列表。"""
        try:
            body = await request.get_json(force=True) or {}
            section = dict(self.plugin.raw_config.get("optimizer_config", {}) or {})
            section["optimizer_provider_id"] = str(body.get("optimizer_provider_id", "")).strip()
            section["vision_provider_id"] = str(body.get("vision_provider_id", "")).strip()
            self.plugin.raw_config["optimizer_config"] = section
            save_fn = getattr(self.plugin.raw_config, "save_config", None)
            if not callable(save_fn):
                return _err("当前 AstrBot 配置对象不支持持久化", 500)
            save_fn()
            return _ok("已生效", **{
                "optimizer_provider_id": section["optimizer_provider_id"],
                "vision_provider_id": section["vision_provider_id"],
            })
        except (TypeError, ValueError) as exc:
            return _err(str(exc))
        except Exception as exc:
            logger.warning(
                "[Imago] PageAPI %s 失败 type=%s", "save_brain_providers", type(exc).__name__,
            )
            return _err("无法保存 Provider 选择", 500)

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
        except Exception as exc:
            logger.warning(
                "[Imago] PageAPI %s 失败 type=%s", "quotas", type(exc).__name__,
            )
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
                prune=True,  # 页面上删掉的行要真的消失，否则填错的 ID 永远清不掉
            )
            return _ok("绘图额度已保存", quotas=rows)
        except (TypeError, ValueError) as exc:
            return _err(str(exc))
        except Exception as exc:
            logger.warning(
                "[Imago] PageAPI %s 失败 type=%s", "save_quotas", type(exc).__name__,
            )
            return _err("无法保存绘图额度", 500)

