"""接口类型 → 可在节点上直接控制的请求参数（插件页渲染下拉框/勾选框）。

这里是唯一的字段定义来源：插件页按 ``api_type`` 取字段渲染控件，保存时再序列化成
``custom_params`` 的 ``key=value`` 文本（适配器只需认识 key，不关心 UI）。

字段结构：
    key       写入 custom_params 的键名
    label     控件标题
    control   select / checkbox / text
    options   select 的取值或 text 的建议值
    default   控件缺省值（保存时会写进 custom_params）
"""

from __future__ import annotations

from typing import Any

PARAM_FIELDS: dict[str, list[dict[str, Any]]] = {
    "openai_image": [
        {
            "key": "size",
            "label": "尺寸",
            "control": "text",
            "placeholder": "1024x1024 / 1:1 / auto / 留空"
        },
        {
            "key": "quality",
            "label": "质量",
            "control": "select",
            "default": "auto",
            "options": [
                "auto",
                "low",
                "medium",
                "high"
            ]
        }
    ],
    "xai_image": [
        {
            "key": "size",
            "label": "比例",
            "control": "text",
            "placeholder": "3:4 / 1024x1024 / auto / 留空"
        },
        {
            "key": "quality",
            "label": "质量",
            "control": "select",
            "default": "medium",
            "options": [
                "low",
                "medium"
            ]
        },
        {
            "key": "nsfw",
            "label": "启用 NSFW",
            "control": "checkbox"
        }
    ],
    "gemini_official": [
        {
            "key": "size",
            "label": "比例",
            "control": "text",
            "placeholder": "16:9 / 1:1 / 留空"
        }
    ],
    "custom_endpoint": []
}
