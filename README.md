# PsyAvatar Care「心澄」

一个可零 GPU 演示的数字人心理支持 Agent：用户像打视频一样与 AI 交流，在明确同意后临时启用摄像头；系统将语音、视觉、评审知识、确定性风险分级和数字人渲染编排成一条可打断、可降级、可审计的会话链路。

> 产品边界：面向成年用户的心理支持与随访演示，不是诊断、治疗、处方或紧急服务。人工接管为模拟流程，不表示已联系真实临床人员。

## 为什么这样融合

- `agent-mind/` 的 Java Agent 没有继续充当主编排核心。其 Agent、记忆、MCP、RAG 概念被重新设计为 Python 3.12 中的显式状态图、类型化工具、评审知识库和追加式事件协议。
- `RAD-NeRF/RAD-NeRF/` 保持只读，保留原 person 222、Wav2Vec 和 RAD-NeRF 资产；通过独立 Python 3.10 worker 与 gRPC 契约接入。
- 原有语音、检索和数字人模型选择保持：`faster-whisper small（CPU INT8）+ Edge TTS Xiaoxiao + BGE-M3/BM25 + Wav2Vec/RAD-NeRF`。本地模式只把文本 Agent 切换为 Ollama 中的 `qwen3.6:latest`；云端 Qwen-Max 与 GPU 数字人仍是需要显式配置的独立边界。

完整分层与回合事件顺序见 [docs/architecture.md](docs/architecture.md)。

## 已实现

- FastAPI + LangGraph 显式 Agent 图
- GREEN / AMBER / RED / EMERGENCY 四级确定性安全路由
- BGE-M3 dense/sparse + BM25 + RRF 混合检索和证据约束
- 工作、情景、语义与隔离安全记忆；敏感长期记忆需同意
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

这些是预录音 PCM 的协议级测量，不是浏览器真实麦克风或临床有效性验证。完整语音补测
后的运行时 SQLite 隐私扫描仍为 `raw_audio=0`、`raw_video=0`、`camera_frame=0`、
`binary_columns=0`。真实设备的收音、扬声器播放与自动播放权限仍需在浏览器中人工确认。
因此当前 CPU-only 大模型结果不应描述为豆包级实时体验；若优先追求通话延迟，应由用户
显式选择更轻量的本地文本模型。

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

- Orchestrator 全部测试、Ruff、严格 mypy
- RAD-NeRF worker 的 CPU 协议测试（GPU smoke 明确跳过）
- Web 单元测试和生产构建
- 固定风险与视觉安全报告

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
