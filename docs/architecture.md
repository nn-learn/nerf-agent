# PsyAvatar Care 工程架构

## 目标边界

这是面向成年用户的数字人心理支持与随访演示，不是诊断、治疗、处方或紧急服务。V1 的人工接管是可审计模拟流程，不会声称已联系真实医生、医院或公共机构。

## 运行时分层

```text
React 通话端
  ├─ 会话 Bearer / 摄像头二次同意
  ├─ Local 浏览器 PCM WebSocket（无 GPU）
  ├─ LiveKit 音视频房间（real）
  └─ Mock 数字人（CI / 静态演示）
          │
          ▼
Python 3.12 Orchestrator
  ├─ 追加式事件流与统一 cancel_token
  ├─ GREEN / AMBER / RED / EMERGENCY 确定性安全路由
  ├─ LangGraph 显式 Agent 图
  ├─ BGE-M3 + BM25 评审知识检索
  ├─ faster-whisper small / Ollama Qwen3.6 / Edge TTS 适配器
  ├─ Qwen3.6-Flash 临时视觉观察
  └─ 脱敏临床演示台与工具审批
          │ gRPC 双向流
          ▼
Python 3.10 Avatar Worker
  ├─ 原 Wav2Vec 音频特征
  ├─ 原 person 222 RAD-NeRF 头部/躯干资产
  ├─ 25 FPS 单调时间戳
  └─ avatar-video LiveKit 轨道
```

Java 项目不再承担实时主链路。它原有的 Agent、记忆、MCP 和 RAG 思路被重新收敛为 Python Orchestrator 中的显式契约；以后若保留 Java 服务，应只通过版本化事件或类型化工具协议接入，避免双编排核心。

## 零 GPU 本地运行面

`ProviderMode=local` 是完整的实时语音演示面，不是 mock 的别名：

- 浏览器采集 16 kHz、单声道、20 ms PCM 帧，经鉴权 WebSocket 进入 FastAPI；
- faster-whisper `small` 在 CPU INT8 上转写；
- Ollama 使用精确配置的 `qwen3.6:latest`，启动时预热；
- BGE-M3 + BM25/RRF 只检索已评审知识，BGE-M3 首次使用可能下载；
- Edge TTS 使用 `zh-CN-XiaoxiaoNeural`，需要网络，失败时降级为文本；
- Docker、LiveKit、Python 3.10 Avatar worker、RAD-NeRF 资产和 CUDA 全部跳过。

本地模式只替换文本 Agent provider，原有 faster-whisper、Edge TTS、BGE-M3
以及 real 路径中的 Wav2Vec/RAD-NeRF 资产均保留。约 23 GB 的
`qwen3.6:latest` 首次加载可能较慢；`qwen:latest` 仅能由用户通过
`PSYAVATAR_TEXT_MODEL` 显式选择，不存在自动静默回退。

## 一次回合

正常回合使用同一 `trace_id`：

```text
transcript.final
→ vision.observation.ready | vision.degraded
→ risk.updated
→ retrieval.completed
→ assistant.response.ready
→ tts.audio.chunk
→ avatar.frame.ready | avatar.degraded
→ playback.started
→ turn.completed
```

插话会在事件锁内使旧 `cancel_token` 失效、清空下游队列并写入 `playback.interrupted`。之后到达的旧语音、视觉结果或数字人帧不能再次进入时间线。

## 隐私与安全

- 原始麦克风音频、摄像头视频和 JPEG 帧只在内存流中存在，事件存储拒绝嵌套二进制。
- 浏览器原始 PCM、原始视频帧和 TTS 音频字节不会持久化或写入日志；持久层会保存受约束且经过审计的文本与元数据事件，包括 `transcript.final` 最终转写、`assistant.response.ready` 助手响应，以及风险、检索、同意、交接、播放/中断、回合与会话生命周期等审计事件。最终转写可能参与模型推理，因此不宣称所有模型输入都是瞬时数据。
- 视觉观察最多 4 帧、10 秒失效，只描述可见对象、文字概况和动作；不能从外观推断诊断、人格、自伤或风险。
- 风险等级由确定性规则拥有，视觉模型和大模型不能自行改写。
- 每个用户会话使用随机 Bearer；LiveKit 令牌只允许向单一房间发布 camera/microphone。
- 临床演示角色必须同时具有 `CLINICIAN_DEMO` 与当前 session scope，只能读取短摘录和脱敏元数据。
- 固定评测覆盖高风险召回、否定/引用误报、视觉诊断边界和 SQLite 媒体扫描。

## 降级顺序

```text
视觉失败 → 语音 + 文本 + 数字人
数字人失败 → 语音 + 文本
TTS 失败 → 文本
STT 失败 → 键盘文字
LiveKit 失败 → HTTP 文字回合
```

默认 mock 模式不依赖后端，适合 CI；local 模式不依赖 Docker、LiveKit 或 GPU，适合真实本地语音演示。真实 RAD-NeRF 仅在独立 Python 3.10/CUDA 环境中显式启用。
