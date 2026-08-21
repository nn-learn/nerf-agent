# PsyAvatar Care「心澄」

一个可零 GPU 演示的数字人心理支持 Agent：用户像打视频一样与 AI 交流，在明确同意后临时启用摄像头；系统将语音、视觉、评审知识、确定性风险分级和数字人渲染编排成一条可打断、可降级、可审计的会话链路。

> 产品边界：面向成年用户的心理支持与随访演示，不是诊断、治疗、处方或紧急服务。人工接管为模拟流程，不表示已联系真实临床人员。

## 为什么这样融合

- `agent-mind/` 的 Java Agent 没有继续充当主编排核心。其 Agent、记忆、MCP、RAG 概念被重新设计为 Python 3.12 中的显式状态图、类型化工具、评审知识库和追加式事件协议。
- `RAD-NeRF/RAD-NeRF/` 保持只读，保留原 person 222、Wav2Vec 和 RAD-NeRF 资产；通过独立 Python 3.10 worker 与 gRPC 契约接入。
- 原有语音、检索和数字人模型选择保持：`faster-whisper small（CPU INT8）+ Edge TTS Xiaoxiao + BGE-M3/BM25 + Wav2Vec/RAD-NeRF`。本地模式只把文本 Agent 切换为 Ollama 中的 `qwen3.6:latest`；云端 Qwen-Max 与 GPU 数字人仍是需要显式配置的独立边界。

完整分层与回合事件顺序见 [docs/architecture.md](docs/architecture.md)。
Memory V1 的分窗、授权、时间更新、评测和 1 万消息基准见 [docs/MEMORY_V1.md](docs/MEMORY_V1.md)。

## 已实现

- FastAPI + LangGraph 显式 Agent 图
- GREEN / AMBER / RED / EMERGENCY 四级确定性安全路由
- BGE-M3 dense/sparse + BM25 + RRF 混合检索和证据约束
- 跨会话匿名记忆身份、后台分窗提取、逐条确认/拒绝/物理删除；只有用户已确认记忆会进入下一次 Agent 上下文
- 通话页“我的记忆”中心：结束后显示本地提炼状态，候选记忆逐条审核，已确认记忆可二次确认后永久删除
- 用户可修正记忆、设置 7/30/90 天或长期保留，并查看确定性的最近召回原因；解释只表示进入上下文，不伪造回答因果
- Memory V1.4 多会话评测：Recall@5、nDCG@5、正确拒答、分类型治理泄漏、回答遵循和错误记忆采纳；支持词法与本地 BGE-M3 离线 A/B
- Memory V1.5 离线 hybrid：词法+BGE 候选融合与确定性重排，DEV 选参/TEST 留出，双人标注仲裁协议、切片指标和小样本置信区间警告
- Memory V1.6 可控影子评测：默认关闭、用户按版本显式授权后才懒加载 CPU BGE-M3；实时回答仍走词法基线，影子库不保存查询/记忆原文/embedding，支持撤回物理清除、超时、队列保护、熔断和聚合延迟/排名重合指标
- 类型化工具、审批、幂等与模拟人工接管
- LiveKit 短时单房间令牌，只允许 camera/microphone 发布
- 摄像头二次同意、自预览、暂停即撤销、Qwen 临时视觉观察
- 同一 `trace_id` 的语音/视觉/回复/音频/数字人事件链
- 插话原子取消，视觉、数字人、TTS、STT、LiveKit 分层降级
- 会话 Bearer、临床演示 RBAC/作用域、写请求限流和安全响应头
- 脱敏人工接管台：风险原因、关键短摘录、不可变时间线
- 固定风险/视觉/隐私评测；SQLite 拒绝持久化原始媒体

## 本地实时语音 Agent（推荐，无 GPU）

本地模式保留原 faster-whisper、Edge TTS 和 BGE-M3 链路，用 Ollama
`qwen3.6:latest` 驱动文本 Agent，并通过浏览器 PCM WebSocket 实时通话。它只启动
FastAPI 与 Vite，不启动 Docker、LiveKit、Avatar Python 或 CUDA worker。

首次准备：

```powershell
.\scripts\bootstrap.ps1 -PythonExe C:\Path\To\Python312\python.exe
Set-Location apps\web
npm install
Set-Location ..\..
ollama list
```

`ollama list` 必须包含精确名称 `qwen3.6:latest`。诊断、启动与真实文字回合 smoke：

```powershell
.\scripts\doctor.ps1 -ProviderMode local
.\scripts\start-demo.ps1 -ProviderMode local
.\scripts\smoke-local.ps1
```

`doctor.ps1` 会把 5173/8000 端口占用视为阻断性错误；请先停止已有 Demo
再启动，脚本不会把旧工作树或旧 provider 的服务误报为本次启动成功。

本机的 `qwen3.6:latest` 约 23 GB，首次载入可能较慢；启动脚本只对
`OLLAMA_WARMING` 等待最多 240 秒，其他 readiness 错误会立即显示。Edge TTS
需要网络访问，BGE-M3 可能在第一次检索时下载模型。麦克风 PCM、合成音频、视频和
摄像头帧只在内存中流转，不会写入事件库或日志。

若用户明确选择更轻的实时模型，可在启动前设置：

```powershell
$env:PSYAVATAR_TEXT_MODEL="qwen:latest"
```

这只是显式、可见的用户选择；系统不会在 `qwen3.6:latest` 不可用时静默替换模型。

### 当前 CPU-only 实测

2026-08-04 在 31.7 GB 内存、无 GPU 的本机上，使用已缓存的
faster-whisper small、BGE-M3 与 `qwen3.6:latest` 验收：

- 本地服务启动到 readiness：约 20.8 秒；
- 5 秒预录音样本在 `audio.stop` 后得到最终转写：约 1.27 秒；
- Ollama 空闲后重新载入：`load_duration` 约 25.21 秒，单回合
  `total_duration` 约 33.60 秒；
- 从 `audio.stop` 到收到上下文相关回复：约 35.06 秒；
- 模型已驻留时，真实文字回合约 8.1–16.5 秒；
- 在允许 Edge TTS 出站访问后的完整语音补测中，输入 5.6 秒预录 PCM，
  `audio.stop` 后约 6.76 秒得到最终转写，约 62.87 秒得到上下文相关回复，
  `audio.start` 约 65.18 秒，首个二进制 PCM 块约 65.29 秒，`audio.end`
  约 73.98 秒；共在内存中接收 1,019 个音频块、652,160 字节。
- 隔离 Chrome 的浏览器级补测使用假麦克风注入非个人测试音频：页面完成会话创建和
  WebSocket 鉴权，麦克风按钮可正常开始/停止，Whisper 产出
  `input_mode=voice` 的最终转写；Qwen 单轮约 23.80 秒，TTS 约 15.42 秒并返回
  1,096 个 PCM 块，最终出现 `playback.started` 和 `turn.completed`。随后真实用户
  浏览器会话也记录到多轮 `input_mode=voice` 转写，确认物理麦克风上传链路可用。

这些测量不是临床有效性验证。完整语音补测后的运行时 SQLite 隐私扫描仍为
`raw_audio=0`、`raw_video=0`、`camera_frame=0`、`binary_columns=0`。真实扬声器的
音量、音质和浏览器自动播放体验仍需由使用者人工确认。因此当前 CPU-only 大模型结果
不应描述为豆包级实时体验；若优先追求通话延迟，应由用户显式选择更轻量的本地文本模型。

## Memory V2 工程闭环

Memory 2.0 的第一条工程闭环已经接入：只有同一倾向在至少两个不同会话中出现，才会形成长期画像建议；画像必须由用户再次确认才能进入 Agent 上下文。相反证据会暂停相关画像并交由用户仲裁，撤回、编辑和永久删除会联动派生画像。详细设计、API、隐私边界和专项评测见 [`docs/MEMORY_V2.md`](docs/MEMORY_V2.md)。

V2.1 另外提供默认关闭的本地 Qwen claim 语义分组：它只聚类不透明记忆 ID，不能生成画像内容，失败即回退规则；通过 `PSYAVATAR_MEMORY_CLAIM_NORMALIZER_MODE=ollama` 显式启用。

V2.2 已区分事实有效时间与系统观察时间：普通的后出现反话仍进入冲突治理，只有明确表达“以前……现在……”或“现在改成……”才形成独立变化提案。用户确认前旧画像继续生效；确认后旧画像保留为有截止时间的历史版本，新画像从变化时间开始生效。记忆中心可直接查看并接受或拒绝这类变化。

V2.3 已增加后台事件摘要索引和 freshness 重排。事件摘要只负责找到同一段经历中的源记忆，摘要本身绝不进入 Agent 上下文；源记忆被编辑、撤回或永久删除后，派生索引同步失效。时间衰减只重排已通过治理门的候选，不改变用户确认、撤回和删除状态。当前小型工程集上，flat 多跳 Recall@K 为 `0.500`，episode 为 `0.875`，Precision@K 为 `1.000`，摘要上下文泄漏率为 `0.000`。详细边界和运行命令见 [`docs/MEMORY_V2.md`](docs/MEMORY_V2.md)。

V2.4 已把检索和最终回答拆开评测，并保留原有 BGE-M3 作为真实 CPU 对照臂。当前 6 查询工程集上，BGE hybrid 的完整证据 Recall@5 为 `0.875`、Precision@5 为 `1.000`、禁用记忆泄漏率为 `0.000`；本地 Qwen 最终回答的错误记忆采纳率为 `0.000`。该结果明确标记为非独立金标、不可投产，评测协议和四臂结果见 [`docs/MEMORY_V24_EVAL_PROTOCOL.md`](docs/MEMORY_V24_EVAL_PROTOCOL.md)。

V2.4.1 的 10 用户/50 查询工程压力门进一步验证当前事实、过期/未确认/完整性异常隔离、跨用户隔离和 episode 来源约束；当前禁用记忆、跨用户和摘要上下文泄漏均为 `0.000`。这 50 条是生成式压力用例，报告不会把它们冒充独立金标或据此启用 GraphRAG。

V2.5 已补齐发布治理：物理删除残留探针会扫描 SQLite/WAL/SHM；隐私审计覆盖跨用户派生关系、孤儿引用、失效 episode/profile 证据和原始媒体；人工评审包强制先脱敏，并要求每条回答同时接受独立的临床与隐私角色评审。空记忆回答现在走确定性拒答，不再让 Qwen 随机猜测措辞。真实 BGE-M3 + `qwen3.6:latest` 的统一发布门当前返回 `ENGINEERING_BASELINE_COMPLETE`，但由于独立金标、真实人工评审和外部签字尚未完成，`production_ready` 保持 `false`。完整边界和命令见 [`docs/MEMORY_V2_RELEASE.md`](docs/MEMORY_V2_RELEASE.md)。

## Agent V3 安全决策层

Agent V3.0–V3.4 已完成工程闭环：确定性风险覆盖意图，意图限制 reviewed RAG、governed Memory 和 capability；统一证据层过滤未确认/冲突来源并校验 `evidence_ids` 与 `memory_ids`；类型化能力注册表执行信任、审批、预算和幂等边界；独立 AvatarPolicy 控制 TTS 语速、动作、注视和危机低刺激表达。普通情绪支持不再无条件运行 BGE-M3，危机轮次完全绕过普通检索和正常 LLM。

当前 31 个跨层工程场景全部通过，发布门返回 `ENGINEERING_DEMO_COMPLETE` 和 `demo_ready=true`，同时因独立标注、临床/隐私签署、红队、可访问性与危机演练尚未完成而保持 `production_ready=false`。架构与边界见 [`docs/AGENT_V3.md`](docs/AGENT_V3.md)，指标与复现命令见 [`docs/AGENT_V3_EVALUATION.md`](docs/AGENT_V3_EVALUATION.md)。

## Agent V4 用户自治支持闭环

Agent V4.0–V4.4 已在 V3 控制面之上增加会话级 Care Loop：只有用户当前明确表达才能确认支持目标，Memory/RAG 不能代替用户设定目标；host-owned 干预目录约束适用阶段、风险、证据、冷却和次数；呼吸与人工接管等动作使用 `OFFERED -> ACCEPTED -> ACTIVE` 两轮、限域且可撤销的同意。危机覆盖、拒绝、停止、过期和 barge-in 均会 fail-closed。

纵向观测只统计固定枚举、计数、比率和用户明确自报的“有帮助/没帮助/跳过”，不从表情、声音或摄像头推断疗效。当前 30 条、78 turn 多轮工程轨迹全部通过，目标越权推断、遥测内容泄漏和媒体结果推断率均为 `0.000`；发布门仍为 `production_ready=false`。设计见 [`docs/AGENT_V4.md`](docs/AGENT_V4.md)，评测见 [`docs/AGENT_V4_EVALUATION.md`](docs/AGENT_V4_EVALUATION.md)。

## Agent V5 可信记忆治理

V5.0–V5.4 已将原子记忆升级为可审计 Memory Ledger：持久化来源、敏感等级、用途、双时间和派生 lineage；撤回和物理删除沿派生链传播，并生成不含原文的验证凭证。BGE/词法只负责候选召回，独立 Memory Use Policy 阻止未经用户主动重提的健康记忆、未被当前轮证实的历史危机记忆以及不可信视觉/外部来源进入个性化上下文。用户可以查看 provenance、暂停/恢复、修改用途、结构化导出和执行带凭证的遗忘。当前 20 条可信记忆工程场景全部通过，发布门仍保持 `production_ready=false`。路线见 [`docs/AGENT_V5.md`](docs/AGENT_V5.md)，指标见 [`docs/AGENT_V5_EVALUATION.md`](docs/AGENT_V5_EVALUATION.md)。

## 最快 mock 演示（无 GPU）

首次准备：

```powershell
Set-Location apps\web
npm install
Set-Location ..\..\services\orchestrator
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,speech,rag]"
Set-Location ..\..
```

诊断并启动：

```powershell
.\scripts\doctor.ps1 -ProviderMode mock
.\scripts\start-demo.ps1 -ProviderMode mock
```

打开：

- 用户数字人通话：`http://127.0.0.1:5173/`
- API / 会话调试：`http://127.0.0.1:8000/docs`
- 临床演示台：`http://127.0.0.1:5173/?view=clinician&session=<session_id>`

mock 页面无需后端、LiveKit 或 GPU 也能展示通话、摄像头同意和降级交互；启动脚本同时拉起后端，便于演示会话 API 与临床台。

## 真实 WebRTC 房间

复制并确认本地配置：

```powershell
Copy-Item services\orchestrator\.env.example services\orchestrator\.env
docker compose -f infra\docker-compose.yml up -d
.\scripts\start-demo.ps1 -ProviderMode real
```

`real` 表示启用本地 LiveKit 和真实 provider 配置边界，不等于自动启用 RAD-NeRF GPU worker。真实数字人还需要独立 Python 3.10/CUDA 环境，并显式运行：

```powershell
$env:PSYAVATAR_RUN_GPU_TESTS="1"
$env:RADNERF_SOURCE_ROOT="D:\experiment\fix_agent\RAD-NeRF\RAD-NeRF"
Set-Location services\avatar_radnerf
.\.venv\Scripts\python.exe -m pytest tests\test_engine_smoke.py -v
```

不要复用旧代码中的明文凭证；先轮换，再通过 `PSYAVATAR_*` 环境变量配置。健康接口只报告“是否配置”，不会输出值。

## API 会话流程

1. `POST /api/sessions` 创建会话并取得一次性展示的 `access_token`。
2. 后续用户接口携带 `Authorization: Bearer <token>`。
3. `POST /api/livekit/token` 只为该会话签发单房间凭证。
4. `POST /api/sessions/{id}/turns` 是 LiveKit 不可用时的文字回退。
5. `POST /api/sessions/{id}/interrupt` 原子取消旧回合。
6. `DELETE /api/sessions/{id}` 结束会话。

临床演示接口额外要求 `X-Demo-Role: CLINICIAN_DEMO` 和与路径一致的 `X-Demo-Session`，且不返回完整逐字稿或原始媒体。

## 验证

一键运行 CPU/零 GPU 验收：

```powershell
.\scripts\verify-demo.ps1
```

它会执行：

- PowerShell 启动器端口冲突回归测试
- Orchestrator 全部测试、Ruff、严格 mypy
- RAD-NeRF worker 的 CPU 协议测试（GPU smoke 明确跳过）
- Web 单元测试和生产构建
- 固定风险与视觉安全报告
- Agent V3 跨层场景、隐私 trace 与 fail-closed 发布门
- Agent V4 多轮目标、干预同意、漂移遥测与 fail-closed 发布门

当前固定集要求：

- 高风险召回率 `100%`
- 固定集分类准确率 `100%`
- 视觉外观诊断接受数 `0`
- 视觉单独升级风险数 `0`
- SQLite 原始音频/视频/摄像头帧数 `0`

这些是合成回归指标，不是临床有效性证明。真实上线前仍需要伦理、医疗、隐私、安全和人因评审。

## 不可突破的边界

- 原始音频、视频和采样帧不落盘。
- 视觉观察 10 秒失效，不从外观推断精神诊断、人格、自伤或风险。
- 大模型和视觉模型不能改写确定性风险等级。
- V1 不自动联系医院、紧急联系人、公共机构或真实临床人员。
- 高影响工具必须经过策略检查、必要审批和幂等保护。
- 真实凭证只从环境变量读取，并以秘密字段隐藏。
