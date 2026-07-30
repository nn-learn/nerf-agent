# PsyAvatar Care 本地 Ollama 实时语音接入设计

日期：2026-07-30  
状态：待用户规格审阅  
目标模型：`qwen3.6:latest`（本机 Ollama，约 23 GB）  
目标环境：Windows、约 31.7 GB 内存、无 GPU

## 1. 背景与目标

当前演示界面已经具备数字人通话外观、摄像头二次同意、会话安全事件、
确定性风险分级和可中断回合等基础设施，但默认 `mock` 模式没有采集麦克风，
后端也没有实例化已经存在的 faster-whisper、Qwen 与 Edge TTS 适配器。

本次工作的目标是打通一个可本地演示、可审计、无 GPU 可运行的真实闭环：

```text
浏览器麦克风
  -> 16 kHz mono PCM
  -> 服务端 VAD 分句
  -> faster-whisper small / CPU INT8
  -> 确定性风险路由 + 经评审检索上下文
  -> 本地 Ollama qwen3.6:latest
  -> AgentResponse 校验与安全闸门
  -> Edge TTS Xiaoxiao
  -> PCM 分块播放 + Web 数字人口型状态
```

首版不追求云产品级延迟，而追求“真实可运行、边界可信、接口可替换、失败可降级”。

## 2. 已确认的产品边界

- 产品定位是面向成年人的非诊断性心理支持演示，不是医疗诊断、治疗、处方或急救服务。
- 风险等级只能由确定性安全路由产生，Qwen、视觉模型和 RAG 均不能改写风险等级。
- 不声称已经联系医生、医院、急救或公共机构。
- 原始麦克风 PCM、摄像头帧和 TTS 音频只在内存流中存在，不写入事件库。
- 摄像头仍需二次同意；本次真实语音接入不自动启用视觉推理。
- 数字人首版继续使用 Web 轻量形象；RAD-NeRF 保留为可选 GPU worker，不阻塞本地演示。
- 原有模型选择保持不变：
  - STT：`faster-whisper small`，CPU，INT8；
  - LLM：本地 Ollama `qwen3.6:latest`；
  - TTS：Edge TTS `zh-CN-XiaoxiaoNeural`。

## 3. 方案选择

### 3.1 采用：浏览器与 FastAPI 直连 WebSocket

首版使用一个会话级 WebSocket 同时承载控制事件、PCM 上行和 TTS PCM 下行。
这避免为单机演示强制引入 Docker 与 LiveKit Agent worker，同时保留现有 LiveKit
代码，后续可把同一套媒体/回合协议迁移到 RTC worker。

### 3.2 未采用的方案

- LiveKit 全链路：更接近正式音视频 RTC，但本地依赖和调试面更大，不适合作为
  无 GPU baseline 的第一条真实链路。
- MediaRecorder + 单次 HTTP 上传：实现简单，但无法稳定实现服务端 VAD、插话取消
  和连续通话状态，不符合“像打视频一样交流”的体验目标。

## 4. 运行时组件

### 4.1 Provider 工厂

应用启动时根据 `PSYAVATAR_PROVIDER_MODE` 创建依赖：

- `mock`：保持当前 CI 和纯 UI 演示行为；
- `local`：
  - `FasterWhisperProvider`；
  - `OllamaTextProvider`；
  - `EdgeTtsProvider`；
  - `MockAvatarClient`；
  - 当前确定性风险、检索和事件存储组件。

`real` 继续预留给未来 LiveKit/RAD-NeRF 部署。配置错误不得静默回落成 mock；
ready health 必须明确报告哪个 provider 未就绪。

### 4.2 OllamaTextProvider

新增原生 Ollama 适配器，调用：

```text
POST http://127.0.0.1:11434/api/chat
```

请求关键字段：

- `model`: `qwen3.6:latest`
- `messages`: 心理支持系统策略 + 当前经裁剪上下文
- `format`: `AgentResponse.model_json_schema()`
- `stream`: 首版为 `false`
- `think`: `false`
- `keep_alive`: 默认 `30m`
- `options.temperature`: 默认 `0.2`
- `options.num_ctx`: 默认 `4096`
- `options.num_predict`: 默认 `320`

首版选择完整结构化响应后再发声，原因是：

1. `AgentResponse` 必须经过 Pydantic schema、确定性风险一致性和 `OutputGuard` 校验；
2. 结构化 JSON 的一部分不能安全地提前播放；
3. 心理支持产品不应在校验完成前把可能越界的半句话说给用户。

Ollama HTTP 响应本身可以在未来切换为 NDJSON 流，但任何语音输出仍需经过句级
安全闸门。该能力作为后续实验开关，不进入首版验收范围。

### 4.3 模型预热与内存策略

`qwen3.6:latest` 约 23 GB，而本机约 31.7 GB 内存。首版采用：

- 后端 ready 检查 `/api/version` 与本地模型列表；
- 第一次真实会话开始时执行一次零内容或最小内容预热；
- `keep_alive=30m`，避免每轮重复装载；
- 同一时间只允许一个本地 LLM 生成任务，避免内存抖动；
- 上下文只保留最近必要回合、结构化摘要和本轮检索证据；
- 超时不自动偷偷换模型，而是向 UI 返回可见的降级原因；
- 配置允许用户显式改成较轻的本地模型，以便对比“质量模式”和“实时模式”。

## 5. WebSocket 协议

端点：

```text
WS /api/sessions/{session_id}/realtime
```

浏览器不能可靠地为 WebSocket 握手添加 Bearer header，因此连接成功后的第一条
消息必须是认证帧；认证前服务端不接受二进制音频。服务端同时校验 `Origin`，
认证超时或失败立即以策略错误关闭。本地演示允许 `ws://127.0.0.1`；
非 loopback 部署必须使用 `wss://`。

### 5.1 客户端 JSON 消息

```json
{"type":"auth","access_token":"pst_..."}
{"type":"audio.start","sample_rate":16000,"channels":1,"encoding":"pcm_s16le"}
{"type":"audio.stop"}
{"type":"turn.interrupt","reason":"user_barge_in"}
{"type":"session.end"}
{"type":"ping","timestamp":0}
```

认证后，麦克风音频以二进制 PCM 帧发送。建议每帧 20 ms，即 640 bytes。
服务端限制单帧大小、每秒帧数、未认证等待时间和单次 utterance 总字节数，
避免用长连接绕过普通 HTTP 限流。

### 5.2 服务端 JSON 事件

```json
{"type":"session.ready","provider_mode":"local","model":"qwen3.6:latest"}
{"type":"agent.state","state":"listening"}
{"type":"transcript.partial","text":"最近我..."}
{"type":"transcript.final","text":"最近我一直睡不好。"}
{"type":"agent.state","state":"thinking"}
{"type":"assistant.response","display_text":"...","risk_level":"GREEN"}
{"type":"audio.start","stream_id":"audio_...","turn_id":"turn_...","sample_rate":16000,"channels":1,"encoding":"pcm_s16le"}
{"type":"agent.state","state":"speaking"}
{"type":"audio.end"}
{"type":"turn.completed","delivery_mode":"voice_text"}
{"type":"error","code":"OLLAMA_TIMEOUT","recoverable":true,"message":"..."}
```

TTS PCM 使用二进制消息下发。`audio.start` 与 `audio.end` 划定一个播放片段，
二者都携带 `stream_id`；同一 WebSocket 上一次只允许一个服务端音频流。
浏览器不得把不属于当前 `turn_id` 的迟到音频加入播放队列。

## 6. 浏览器媒体链路

新增 `useLocalRealtimeMediaSession`：

1. 创建后端会话并保存仅存在内存中的 access token；
2. 建立 WebSocket 并发送认证帧；
3. 用户开启麦克风时调用带回声消除、降噪和自动增益约束的
   `getUserMedia({audio: ...})`；
4. 使用 AudioWorklet 将浏览器采样率重采样为 16 kHz、mono、PCM S16LE；
5. 按 20 ms 发送音频；
6. 接收状态、字幕和二进制 TTS PCM；
7. 使用 Web Audio 顺序播放 PCM，并以播放队列驱动数字人 `speaking` 状态；
8. 用户插话时立即停止本地播放、清空队列并发送 `turn.interrupt`。

页面新增动态字幕来源：

- 用户最终转写；
- 小澄经过安全校验的 `display_text`；
- provider 错误或降级说明。

键盘文本输入作为 STT 失败时的必要降级能力保留，并复用同一回合处理链。

## 7. VAD、转写与插话

- 输入格式固定为 16 kHz mono PCM S16LE；
- WebRTC VAD 使用 20 ms 帧；
- 保留约 300 ms speech pre-roll；
- 连续静音约 600 ms 结束一次 utterance；
- 单次 utterance 默认上限 20 秒，超限后强制切段；
- 空白或低可信转写不触发 LLM；
- agent 正在生成或播放时检测到新语音：
  - 立即使旧 `cancel_token` 失效；
  - 停止浏览器播放；
  - 取消 Ollama HTTP task、TTS task 和 avatar task；
  - 新语音进入下一回合。

取消的任务即使稍后返回，也不能写入当前时间线或重新开始播放。

## 8. 安全与 RAG

处理顺序固定为：

```text
转写
-> 确定性风险识别
-> 风险路由
-> reviewed-only 检索
-> Qwen 生成
-> AgentResponse schema 校验
-> risk_level 一致性校验
-> 诊断/急救声称/证据引用输出闸门
-> TTS
```

RED/EMERGENCY 路由必须能够绕开普通生成模板，优先返回经过评审的稳定话术和
人工接管建议。视觉观察不能单独升级或降级风险。模型给出的 `evidence_ids`
必须是本轮检索候选的子集，否则响应拒绝并进入安全降级话术。

首版不扩大“记忆”范围：只使用现有会话内摘要与显式候选，不将原始转写自动写成
跨会话长期记忆。任何长期记忆写入需要后续单独的同意、删除和审计设计。

## 9. 错误与降级

| 故障 | 用户可见行为 | 系统行为 |
|---|---|---|
| Ollama 未启动 | 显示“本地模型未连接” | ready=false，不进入假 connected |
| 模型不存在 | 显示精确模型标签 | 不自动下载或替换 |
| 模型加载/生成超时 | 保留转写，提示可重试或切轻量模型 | 取消生成并记录脱敏事件 |
| STT 不可用 | 提供键盘输入 | 不伪造转写 |
| Edge TTS/网络不可用 | 只显示文本 | 不影响已校验回答 |
| WebSocket 断开 | 停止采集和播放 | 可重连但不重放旧音频 |
| 安全校验失败 | 使用固定安全降级话术 | 记录 guard 原因，不保存原始媒体 |

## 10. 配置

新增或调整的本地环境变量：

```dotenv
PSYAVATAR_PROVIDER_MODE=local
PSYAVATAR_OLLAMA_BASE_URL=http://127.0.0.1:11434
PSYAVATAR_TEXT_MODEL=qwen3.6:latest
PSYAVATAR_OLLAMA_KEEP_ALIVE=30m
PSYAVATAR_OLLAMA_TIMEOUT_SECONDS=180
PSYAVATAR_OLLAMA_NUM_CTX=4096
PSYAVATAR_OLLAMA_NUM_PREDICT=320
PSYAVATAR_STT_MODEL=small
PSYAVATAR_TTS_VOICE=zh-CN-XiaoxiaoNeural
VITE_MEDIA_MODE=local
```

模型加载可能超过普通 HTTP 超时，因此 load/first-turn 超时与 steady-state
生成超时需要分开观测；首版可以共享较宽的 180 秒上限，但 UI 必须持续显示状态。

## 11. 可观测性

每轮继续使用同一 `trace_id` 和 `cancel_token`。新增仅存元数据：

- VAD utterance 时长；
- STT 时长；
- Ollama load、prompt-eval、generation 与总时长；
- 输出 token 数；
- TTS 首块与总时长；
- 是否中断、超时或降级。

日志和 SQLite 事件均不得包含 access token、原始 PCM、原始视频帧或 TTS 二进制。

## 12. 测试与验收

### 12.1 自动化测试

- Ollama provider 请求格式、schema 校验、风险不可改写、超时和取消；
- provider 工厂在 `mock`/`local` 下的装配；
- WebSocket 认证、Origin、认证前拒绝音频、断开清理；
- VAD 分段、最长 utterance、静音结束；
- 插话后旧 token 的音频和响应不可到达客户端；
- 二进制媒体不进入 EventStore；
- 前端状态转换、字幕更新、PCM 队列、interrupt 清空；
- 原有安全、风险召回、视觉边界和 clinician 测试不回归。

### 12.2 本机集成验收

1. `ollama list` 可见 `qwen3.6:latest`；
2. ready health 明确报告 local providers；
3. 用户说一句普通中文，页面出现真实转写；
4. Qwen 返回非 mock 的上下文相关回答；
5. 页面显示回答，并由 Edge TTS 播放；
6. 数字人在音频播放期间进入 speaking 状态；
7. 用户点击打断或再次说话，旧音频立即停止；
8. 高风险测试语句进入确定性安全路由；
9. 关闭 Ollama 后页面明确报错，不假装已回答；
10. SQLite 扫描确认没有原始媒体。

## 13. 实施顺序

1. Ollama provider、配置和 provider 工厂；
2. 本地文本回合的真实模型集成测试；
3. WebSocket 会话协议与服务端 VAD/STT/TTS；
4. 浏览器 AudioWorklet、播放队列和状态机；
5. 插话取消与降级；
6. 本机端到端验证、延迟记录和文档更新。

本设计不会删除现有 mock 或 LiveKit 路径，避免破坏 CI 和未来 RTC 演进空间。

## 14. 依据

- Ollama Chat API：<https://docs.ollama.com/api/chat>
- Ollama Structured Outputs：<https://docs.ollama.com/capabilities/structured-outputs>
- Ollama Streaming：<https://docs.ollama.com/api/streaming>
- Ollama keep-alive FAQ：<https://docs.ollama.com/faq>
