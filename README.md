# IMAGO·映相

IMAGO·映相是 AstrBot 的异步图片生成插件，支持普通绘图、Persona 出镜、参考图、多节点/多模型 fallback、绘图额度和素材管理。

- 版本：`1.0.2`
- AstrBot：`>=4.16,<5`

## 安装

将 `imago` 放入 AstrBot 插件目录并加载。依赖：

```text
aiohttp>=3.9,<4
```

运行数据写入 AstrBot 分配的 `plugin_data/imago/`，不写入源码目录。

## 节点配置

至少配置一个有效节点：节点 ID、接口类型、Base URL、API Key、模型、超时和默认尺寸。API Key 可逐行填写多个，每个节点独立按调用顺序轮换，不随机挑选。

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
| `/imago help` | 查看按用途和当前权限分组的帮助卡片；渲染失败时回退纯文本 |
| `/imago draw <提示词>`、`/画 <提示词>` | 普通绘图 |
| `/imago photo <画面要求>`、`/拍照 <画面要求>` | 当前 Persona 出镜 |
| `/imago status` | 查看自己的任务 |
| `/imago quota help/show/sign` | 查看、查询或签到 |
| `/imago quota add/del/set <用户 ID> <整数>` | 管理员调额 |
| `/imago ref-upload` | 上传 Persona 参考图 |
| `/imago summary-show/rebuild/set` | 查看、重建或设置外观摘要 |
| `/imago provider-primary <节点 ID>` | 指定主节点（也可在 WebUI 设置） |

`/imago photo` 与 `/拍照` 的受理回复使用“当前人设”，不显示英文 `Persona`。

`/imago draw`、`/画`、`/imago photo`、`/拍照` 的尾部提示词/画面要求使用 `GreedyStr` 语义，描述包含空格时不会被截断为第一个词。

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
- Persona 识图 Provider 只在 Plugin Page 手动勾选图片并重建摘要时使用，且必须支持图片输入。
- 任务会读取当前消息、引用消息和正文中的参考图，来源支持本地路径、HTTP(S)、`data:` 与 `base64://`。引用消息图片在事件处理阶段前台解析并并入任务参考图：只有引用 id 且无正文、或正文含 `[图片]` 等占位时按严格模式处理，提取不到图片会返回“引用消息图片无法获取”；明确为纯文字引用时保持宽松。
- 参考图在事件阶段完成本地化接管与去重，无法解析时任务严格失败，不会静默退化为纯文生图；远程图下载后同样按文件魔数识别格式，空响应或内容不是图片会严格失败，头部 MIME 与内容不一致时以内容魔数为准。
- 任务输入、输出和去身份化 manifest 位于 `plugin_data/imago/task_cache/<task_id>/`。
- Persona 出镜默认采用自然第三方视角（他拍观感）：平视或轻微俯仰的中景/全景，仿佛画面外的摄影师拍摄，避免默认怼脸自拍或特写；用户明确要求自拍、特写或指定机位/视角时始终以用户为准。副脑关闭且未启用降级注入时，视角由用户描述决定。

### 副脑优化原理

- 只处理 Persona 出镜任务；普通绘图不调用副脑。
- 开启后，插件把稳定外观摘要与本轮画面要求分开交给副脑。副脑只重写本轮动作、场景、临时服饰、视角、构图、镜头和光线，不改写稳定外观。
- 副脑 system prompt 依次由“副脑自定义提示词”、可选风格预设和插件固定协议组成。用户当轮明确指定的风格、媒介、视角和构图优先。
- `None(无)` 不追加任何内置风格；`default(通用)`、写实、电影感、动漫和 3D 会追加对应的英文视觉指导。
- 插件固定协议仍负责隔离外观摘要、限定输出格式并保留画面文字原文，因此 `None(无)` 只关闭风格预设，不关闭这些必要约束。
- 关闭副脑优化后，不调用副脑 Provider，直接把原始画面描述作为动态部分，与稳定外观摘要组合后交给图片模型。
- 副脑正常完成时，风格预设与默认第三方视角由副脑消化，最终 prompt 只做纯拼接，不重复注入后缀。
- “副脑降级时注入风格后缀”开关（默认关闭）：启用后，副脑关闭或调用失败降级时，将所选风格预设、副脑自定义提示词与默认第三方视角以低优先级英文后缀直接写入最终 prompt；关闭时降级路径保持原始描述。副脑调用失败不再终止任务，改用原始画面描述继续生成。
- 用户明确指定的风格、视角、机位、自拍或特写始终优先于上述预设。

## 任务、页面与日志

`generate_image` 和 `generate_persona_image` 只创建后台任务；成功返回不代表已生成或平台已确认送达。Plugin Page 提供 Persona、图片节点和额度三个管理区域；额度表可按任意列升降序排列，只维护稳定的用户 ID，不缓存昵称。

任务进度只通过 `/imago status` 查询。插件不会向主 LLM 注入后台任务状态、阶段或最近任务信息。

至少一张图片成功落盘才算生成成功；Provider 成功响应却没有可识别图片，或所有结果均无法落盘时，任务为 `no_output`，仍可继续模型 fallback，但最终不按普通失败退款。失败通知是否发送成功会单独写入 manifest。主动发送会补跑装饰链，兼容 BubbleReply 这类通过 `event.send()` 发前置分段的插件；任一前置分段失败时仍尽量发送剩余主链，但整次投递不会误记成功。

`task_config.llm_caption`（默认关闭）开启后，出图成功或失败时会用会话 Chat Provider 按当前人设生成一句简短配文与结果一起发送，配文在前、图片拼接在其后。`llm_caption_cm_context`（默认关闭）开启后配文请求复用 ChatMemory 的接管上下文（只通过公开实例 API 只读查询，模式同 time_awareness），关闭时仅携带任务局部上下文。`llm_caption_pregen`（默认关闭）开启后，图片生成期间并行预生成成功版通用配文，出图成功直接使用；任务失败时预生成结果丢弃，失败通知走同步配文。LLM 调用失败或时间预算不足时自动回退固定文案。

配置修改后对后续请求动态生效。并发上限提高会放行等待任务，降低时不取消已运行任务；单个已开始的 Provider 尝试仍使用其创建时的节点参数快照。

DEBUG 会记录阶段、节点和具体模型、成功/失败、参考图数量、脱敏后的提示词与参数、结果落盘和发送阶段；不会记录密钥、完整 base64、原始 URL 或绝对路径。日志等级请到 WebUI 插件详情页调整（运行期生效）；`log_with_bot_id` 开启后日志前缀附加机器人实例 ID，便于多 Bot 排查。

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

- 默认阻止回环、局域网和其他非公网图片 URL（含公网 URL 重定向到内网的情况），不新增白名单；被拒绝时请改用公网 URL，或确认风险后关闭 `storage_config.block_private_networks`。
- 远程图、`data:`/`base64://` 与 `/imago ref-upload` 共用同一套来源分类、SSRF 校验与大小限制；`/imago ref-upload` 单次最多上传 20 张。
- Persona 数据按安全编码的 ID 隔离，参考图按哈希去重。
- 摘要和额度采用原子写入，任务缓存按任务目录隔离；插件重载时取消运行任务。

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s imago/test -v
node --check imago/pages/webui/app.js
```

真实 Provider、平台引用消息和 Plugin Page 交互仍需在目标 AstrBot 环境验收。
