# Imago · 映相

Imago 是 AstrBot 的异步图片生成插件，支持普通绘图、Persona 出镜、参考图、多节点/多模型 fallback、绘图额度和素材管理。

- 版本：`1.0.0`
- AstrBot：`>=4.16,<5`
- 完整架构、异步任务状态机、发送语义和 Review 清单：[ARCHITECTURE.md](ARCHITECTURE.md)

## 安装

将 `imago` 放入 AstrBot 插件目录并加载。依赖：

```text
aiohttp>=3.9,<4
```

运行数据写入 AstrBot 分配的 `plugin_data/imago/`，不写入源码目录。

## 节点配置

至少配置一个有效节点：节点 ID、接口类型、Base URL、API Key、模型、超时和默认尺寸。API Key 可逐行填写多个，插件按请求轮换使用。

调用顺序：

```text
WebUI 选择的主节点默认模型 → 主节点备选模型 → 其他节点默认模型 → 其他节点备选模型
```

在 Plugin Page（WebUI）节点列表点击左侧图标并保存，即可切换主节点。未设置或所选节点无效时，第一个有效节点为主节点。同一节点的默认模型和备选模型必须使用同一协议与 Base URL；共用百炼 multimodal-generation 契约的 WAN 与 Qwen 可以放在同一节点。

### 接口类型

| 类型 | 地址示例 | 协议/参考图 |
| --- | --- | --- |
| `openai_image` | `https://api.example.com/v1` | images API / multipart |
| `openai_chat` | `https://api.example.com/v1` | chat/completions / base64 |
| `gemini_official` | `https://generativelanguage.googleapis.com/v1beta` | generateContent / `inlineData` |
| `dashscope_multimodal` | 百炼完整 generation URL | MultiModal / data URL |
| `custom_endpoint` | 自定义完整 POST URL | Imago JSON / base64 |

有参考图时自动走图生图分支；否则走文生图。

### 参考图总上限

`reference_image_limit` 是本次请求的总参考图上限：消息中明确附带的图片优先占用名额，剩余名额再随机抽取 Persona 固定图。例如上限为 `3` 且用户带 `1` 张图时，最多取 `2` 张 Persona 图；`0` 表示不限制。节点的默认模型和备选模型共用此值。用户附图超过上限时不会丢弃，Provider 可能按自身限制拒绝请求。

## 命令

| 命令 | 作用 |
| --- | --- |
| `/imago help` | 查看命令 |
| `/imago draw <提示词>`、`/画 <提示词>` | 普通绘图 |
| `/imago photo <画面要求>`、`/拍照 <画面要求>` | 当前 Persona 出镜 |
| `/imago status` | 查看自己的任务 |
| `/imago quota help/show/sign` | 查看、查询或签到 |
| `/imago quota add/del/set <用户 ID> <整数>` | 管理员调额 |
| `/imago ref-upload` | 上传 Persona 参考图 |
| `/imago summary-show/rebuild/set` | 查看、重建或设置外观摘要 |
| `/imago provider-primary <节点 ID>` | 指定主节点（也可在 WebUI 设置） |

`/imago photo` 与 `/拍照` 的受理回复使用“当前人设”，不显示英文 `Persona`。

## 绘图额度

- 黑名单禁止绘图指令，并从本轮 LLM 请求移除两个 Imago 绘图工具；任务入口仍会复查。
- 无限额度白名单不扣额度且无日上限；黑名单优先。
- 关闭 `enable_quota` 时普通用户不扣额度，但黑名单仍生效。
- 开启后按请求图片数量扣费；scheduler 拒绝入队会立即退款，后台任务仅在终态为 `failed` 时退回本次实际扣费。`no_output`、超时及其他终态均不退款。
- 开启签到后，每人每天可 `/imago quota sign` 一次，获得配置范围内的随机额度。

`quota_config.daily_quota_target` 是每日目标余额。当天首次访问额度时精确重置到该值，低补高削；关闭 `enable_daily_refresh` 则不刷新。余额保存在 `plugin_data/imago/quotas.json`，也可在 Plugin Page 的额度页批量编辑。

## Persona 与参考图

- 普通绘图直接使用主 LLM 的完整画面描述；Persona 图片会自动加入稳定外观摘要和本轮动态描述。
- 手工摘要优先；自动摘要会随 Persona Prompt 变化失效，使用参考图生成的摘要还会在所选图删除后失效。
- 只有在页面手动选择图片并重建摘要时，图片才会交给识图 Provider。
- 任务会读取当前消息、引用消息和正文中的公网图片。明确参考图无法读取时，不会静默退化为纯文生图。
- 任务输入、输出和去身份化 manifest 位于 `plugin_data/imago/task_cache/<task_id>/`。

## 任务、页面与日志

`generate_image` 和 `generate_persona_image` 只创建后台任务；成功返回不代表已生成或平台已确认送达。Plugin Page 提供 Persona、图片节点和额度三个管理区域。

任务进度只通过 `/imago status` 查询。插件不会向主 LLM 注入后台任务状态、阶段或最近任务信息。

至少一张图片成功落盘才算生成成功；Provider 成功响应却没有可识别图片，或所有结果均无法落盘时，任务为 `no_output`，仍可继续模型 fallback，但最终不按普通失败退款。失败通知是否发送成功会单独写入 manifest。主动发送会补跑装饰链，兼容 BubbleReply 这类通过 `event.send()` 发前置分段的插件；任一前置分段失败时仍尽量发送剩余主链，但整次投递不会误记成功。

配置修改后对后续请求动态生效。并发上限提高会放行等待任务，降低时不取消已运行任务；单个已开始的 Provider 尝试仍使用其创建时的节点参数快照。

DEBUG 会记录阶段、节点和具体模型、成功/失败、参考图数量、脱敏后的提示词与参数、结果落盘和发送阶段；不会记录密钥、完整 base64、原始 URL 或绝对路径。`debug_to_info` 可将插件 DEBUG 提升为 INFO。

## `custom_endpoint` 契约

向配置的完整 URL 发送 `Authorization: Bearer <api_key>` JSON POST：

```json
{
  "prompt": "完整提示词",
  "model": "model-id",
  "count": 1,
  "size": "1024x1024",
  "aspect_ratio": "1:1",
  "references": [{"mime_type": "image/png", "data": "BASE64"}],
  "parameters": {"quality": "high"}
}
```

响应使用 `{"data":[{"url":"..."}]}` 或 `{"data":[{"b64_json":"..."}]}`。`parameters` 只接受用户通过 `--key value` 传入的值。

## `dashscope_multimodal`

```text
api_type: dashscope_multimodal
base_url: https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation
model: qwen-image-3.0-pro
default_size: 1024x1024
```

Base URL 必须是同地域、可直接 POST 的完整 generation URL。插件会构造单轮 `input.messages`，把尺寸中的 `x` 转成 `*`，默认开启 `prompt_extend`，并支持 `negative_prompt`、`seed`、`watermark` 等 `extra_params`。

插件不按模型名写死图片数量或大小能力。参考图数量由节点的 `reference_image_limit` 控制，用户明确附图不会被静默删除；模型自身限制由百炼响应决定。百炼错误会保留脱敏后的 `code/message` 并进入可用的模型/节点 fallback。

## 安全与验证

- 默认阻止回环、局域网和其他非公网图片 URL。
- Persona 数据按安全编码的 ID 隔离，参考图按哈希去重。
- 摘要和额度采用原子写入，任务缓存按任务目录隔离；插件重载时取消运行任务。

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s imago/test -v
node --check imago/pages/webui/app.js
```

主动发送与 BubbleReply 的无平台 smoke 测试见 `test/runtime_active_send_smoke.py`。真实 Provider、平台引用消息和 Plugin Page 交互仍需在目标 AstrBot 环境验收。
